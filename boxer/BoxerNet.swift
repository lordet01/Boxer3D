// BoxerNet.swift
// Apple CoreML 기반 BoxerNet 추론 래퍼.
//
// 모델 변환 파이프라인 (한 번만 실행해두면 됨):
//   1) python scripts/strip_if_nodes.py
//        --input  models_backup/BoxerNet.fp32.onnx
//        --output models_backup/BoxerNet.fp32.no_if.onnx
//   2) python scripts/convert_to_coreml.py
//        --input  models_backup/BoxerNet.fp32.no_if.onnx
//        --output boxer/BoxerNet.mlpackage
//        --num-boxes 8 --precision fp16 --target ios16
//
// 그 결과 `boxer/BoxerNet.mlpackage` 가 생성되며 이 파일을 Xcode 번들에 추가해
// `Bundle.main.url(forResource: "BoxerNet", withExtension: "mlpackage")` 로 찾는다.
//
// 왜 ONNX Runtime + CoreML EP 가 아닌 CoreML 직접 변환인가:
//  * ORT의 CoreML EP는 ONNX 그래프의 dtype을 그대로 따라가서 ViT의 attention
//    경로에서 GPU FP16 overflow → NaN 을 자주 뱉었다.
//  * Apple CoreML 컨버터는 transformer-aware한 dtype 정책으로
//    LayerNorm/Softmax/MatMul 주변을 자동으로 안전하게 다룬다.
//  * .mlpackage 가 자체 압축으로 ONNX 보다 훨씬 작다 (BoxerNet: 400 MB → 200 MB).
//  * iOS 의 ANE/GPU/CPU 디스패치가 CoreML에 가장 잘 통합돼 있다.

import Foundation
import Accelerate
import simd
import CoreML

// MARK: - Data Types

/// A single 3D bounding box detection.
struct Detection3D {
    /// Centre position in ARKit world coordinates (metres).
    let center: simd_float3
    /// Box dimensions (width, height, depth) in metres.
    let size: simd_float3
    /// Yaw angle in radians [-pi/2, pi/2].
    let yaw: Float
    /// Detection confidence [0, 1].
    let confidence: Float
    /// Full 4x4 transform for placing in ARKit scene.
    let worldTransform: simd_float4x4
    /// Object class label from YOLO.
    let label: String?
}

/// A 2D bounding box from YOLO in pixel coordinates.
struct Box2D {
    let xmin: Float
    let ymin: Float
    let xmax: Float
    let ymax: Float
    var label: String? = nil
    var score: Float = 0
}

// MARK: - BoxerNet

final class BoxerNet {
    private let model: MLModel

    /// Image size the model expects (960x960).
    static let imageSize: Int = 960
    /// Patch size used by DINOv3.
    static let patchSize: Int = 16
    /// Feature grid dimensions.
    static let gridH: Int = imageSize / patchSize  // 60
    static let gridW: Int = imageSize / patchSize  // 60
    static let numPatches: Int = gridH * gridW     // 3600

    /// CoreML 모델은 변환 시점에 `num_boxes` 차원을 정수로 박았기 때문에
    /// 호출부는 항상 정확히 이 값만큼의 박스를 보내야 한다.
    /// 부족하면 더미 박스(score=0)로 padding 한다 (`predict` 안에서 처리).
    /// 변환 시 `--num-boxes 8` 와 일치해야 한다.
    static let fixedNumBoxes: Int = 8

    /// CoreML 컴퓨트 유닛 선택.
    /// `.all` = ANE/GPU/CPU 자동 디스패치 (가장 빠르고 메모리 효율적).
    /// 디버깅 목적으로 `.cpuOnly` / `.cpuAndGPU` 로 바꿀 수 있다.
    static let preferredComputeUnits: MLComputeUnits = .all

    init(modelPath: String) throws {
        // .mlpackage 또는 .mlmodelc 둘 다 받을 수 있게 한다.
        // 번들에 .mlpackage 가 들어 있으면 iOS 빌드 시 자동으로 .mlmodelc 로 컴파일됨.
        let url = URL(fileURLWithPath: modelPath)

        let configuration = MLModelConfiguration()
        configuration.computeUnits = Self.preferredComputeUnits

        let loadStart = CFAbsoluteTimeGetCurrent()
        // .mlpackage 는 MLModel(contentsOf:) 가 직접 받는다 (iOS 16+).
        // 첫 로드 시 컴파일이 일어날 수 있어 시간이 걸리고, 컴파일 결과는
        // 디스크에 캐시된다.
        self.model = try MLModel(contentsOf: url, configuration: configuration)
        let loadMs = (CFAbsoluteTimeGetCurrent() - loadStart) * 1000

        let sizeBytes: Int = {
            if url.hasDirectoryPath {
                // mlpackage 는 디렉토리 — 안의 모든 파일 합산.
                let enumerator = FileManager.default.enumerator(at: url,
                                                                includingPropertiesForKeys: [.fileSizeKey])
                var total = 0
                while let f = enumerator?.nextObject() as? URL {
                    let sz = (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                    total += sz
                }
                return total
            } else {
                return ((try? FileManager.default.attributesOfItem(atPath: modelPath)[.size] as? NSNumber)?.intValue ?? 0)
            }
        }()
        let sizeMB = sizeBytes / 1_000_000

        let cu: String
        switch Self.preferredComputeUnits {
        case .cpuOnly: cu = "cpuOnly"
        case .cpuAndGPU: cu = "cpuAndGPU"
        case .all: cu = "all (ANE+GPU+CPU)"
        case .cpuAndNeuralEngine: cu = "cpuAndNeuralEngine"
        @unknown default: cu = "?"
        }
        print("[boxer] BoxerNet loaded: \(url.lastPathComponent) (\(sizeMB) MB), "
              + "computeUnits=\(cu), num_boxes fixed=\(Self.fixedNumBoxes), "
              + String(format: "load=%.0fms", loadMs))
    }

    /// 더미 입력 1회 추론 → CoreML의 graph 컴파일/캐시 비용을 앱 시작 시점으로 옮김.
    /// 두 번째 호출부터는 캐시 적중으로 거의 즉시 시작한다.
    func warmup() throws {
        let S = Self.imageSize
        let N = Self.numPatches
        let M = Self.fixedNumBoxes
        let dummyImage = [Float](repeating: 0, count: 3 * S * S)
        let dummySDP = [Float](repeating: 1.0, count: N)
        let dummyBB = [Float](repeating: 0.5, count: M * 4)
        let dummyRay = [Float](repeating: 0, count: N * 6)
        _ = try runInference(image: dummyImage, sdpPatches: dummySDP,
                             bb2d: dummyBB, rayEncoding: dummyRay, numBoxes: M)
    }

    // MARK: - Public API

    /// Run full pipeline: preprocess, infer, postprocess.
    ///
    /// - Parameters:
    ///   - image: RGB pixel data resized to 960x960, float32 in [0, 1], shape (3, 960, 960) in CHW.
    ///   - depthMap: LiDAR depth map from ARKit (metres). Can be smaller than 960x960.
    ///   - intrinsics: 3x3 camera intrinsics matrix from ARFrame.camera.intrinsics.
    ///   - cameraTransform: 4x4 world transform from ARFrame.camera.transform.
    ///   - boxes2D: Array of YOLO 2D detections in pixel coords (for the 960x960 image).
    ///   - confidenceThreshold: Minimum confidence to keep a detection.
    /// - Returns: Array of 3D detections in world coordinates.
    func predict(
        image: [Float],            // CHW flat array, len = 3 * 960 * 960
        depthMap: [[Float]],       // HxW depth in metres (0 = invalid)
        intrinsics: simd_float3x3, // fx, fy, cx, cy from ARKit
        cameraTransform: simd_float4x4,
        boxes2D realBoxes2D: [Box2D],
        confidenceThreshold: Float = 0.3
    ) throws -> [Detection3D] {
        guard !realBoxes2D.isEmpty else { return [] }

        // 모델 입력은 정확히 `fixedNumBoxes`개여야 한다 (CoreML이 정수 dim으로 박힘).
        // 부족하면 image 중앙의 작은 더미 박스를 padding(score=0). 출력에서는
        // realCount 미만 인덱스만 살린다.
        let M = Self.fixedNumBoxes
        let realCount = min(realBoxes2D.count, M)
        var boxes2D = Array(realBoxes2D.prefix(realCount))
        if boxes2D.count < M {
            let pad = Box2D(
                xmin: Float(Self.imageSize / 2),
                ymin: Float(Self.imageSize / 2),
                xmax: Float(Self.imageSize / 2 + 1),
                ymax: Float(Self.imageSize / 2 + 1),
                label: nil, score: 0
            )
            boxes2D.append(contentsOf: Array(repeating: pad, count: M - boxes2D.count))
        }

        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[2][0]
        let cy = intrinsics[2][1]

        // 1. Convert ARKit camera (OpenGL: -Z forward, Y up) to
        //    OpenCV convention (Z forward, Y down) that BoxerNet expects.
        //    Flip Y and Z axes of the camera local frame.
        let flipYZ = simd_float4x4(columns: (
            simd_float4( 1,  0,  0, 0),
            simd_float4( 0, -1,  0, 0),
            simd_float4( 0,  0, -1, 0),
            simd_float4( 0,  0,  0, 1)
        ))
        let T_wc = cameraTransform * flipYZ

        // Gravity-aligned voxel frame.
        let T_wv = gravityAlign(T_worldCam: T_wc)
        let T_vc = T_wv.inverse * T_wc

        // 2. Build SDP patches (median LiDAR depth per 16x16 patch).
        let sdpPatches = buildSDPPatches(
            depthMap: depthMap,
            fx: fx, fy: fy, cx: cx, cy: cy
        )

        // 3. Normalise 2D boxes.
        let W = Float(Self.imageSize)
        let H = Float(Self.imageSize)
        var bb2dFlat: [Float] = []
        for box in boxes2D {
            bb2dFlat.append((box.xmin + 0.5) / W)
            bb2dFlat.append((box.xmax + 0.5) / W)
            bb2dFlat.append((box.ymin + 0.5) / H)
            bb2dFlat.append((box.ymax + 0.5) / H)
        }

        // 4. Compute Plucker ray encoding.
        let rayEncoding = buildRayEncoding(T_vc: T_vc, fx: fx, fy: fy, cx: cx, cy: cy)

        // 5. Run CoreML inference.
        // 입력 통계도 같이 찍어 입력 자체가 깨졌는지 즉시 확인 가능하게 한다.
        Self.logTensorStats("image", image)
        Self.logTensorStats("sdp",   sdpPatches)
        Self.logTensorStats("bb2d",  bb2dFlat)
        Self.logTensorStats("ray",   rayEncoding)

        let (centers, sizes, yaws, confidences) = try runInference(
            image: image,
            sdpPatches: sdpPatches,
            bb2d: bb2dFlat,
            rayEncoding: rayEncoding,
            numBoxes: M
        )

        // 진단: 실제 모델이 무엇을 반환했는지 한 번에 보이도록 통계를 찍는다.
        let realConfs = Array(confidences.prefix(realCount))
        let nanCount = realConfs.filter { $0.isNaN }.count
        let infCount = realConfs.filter { $0.isInfinite }.count
        let validConfs = realConfs.filter { $0.isFinite }
        let cMin = validConfs.min() ?? .nan
        let cMax = validConfs.max() ?? .nan
        let cMean = validConfs.isEmpty ? Float.nan : validConfs.reduce(0, +) / Float(validConfs.count)
        let centersHead = (0..<min(realCount, 2)).map { i in
            String(format: "(%.2f,%.2f,%.2f)", centers[i*3], centers[i*3+1], centers[i*3+2])
        }.joined(separator: ", ")
        print(String(format:
            "[boxer] raw confs (N=%d): min=%.3f max=%.3f mean=%.3f  NaN=%d  Inf=%d  centers[0..2]=[\(centersHead)]",
            realCount, cMin, cMax, cMean, nanCount, infCount))

        // 6. Postprocess: voxel → world coords. realCount 이후는 padding이라 건너뛴다.
        var detections: [Detection3D] = []
        for i in 0..<realCount {
            let conf = confidences[i]
            guard conf.isFinite, conf >= confidenceThreshold else { continue }

            let centerVoxel = simd_float3(centers[i * 3], centers[i * 3 + 1], centers[i * 3 + 2])
            let size = simd_float3(sizes[i * 3], sizes[i * 3 + 1], sizes[i * 3 + 2])
            let yaw = yaws[i]

            // Transform centre: voxel → world.
            let centerWorld = (T_wv * simd_float4(centerVoxel, 1.0)).xyz

            // Build world rotation: T_world_voxel.R * R_yaw.
            let R_wv = upperLeft3x3(T_wv)
            let R_yaw = rotationZ(angle: yaw)
            let R_world = R_wv * R_yaw

            // Build 4x4 transform for ARKit placement.
            var transform = simd_float4x4(1.0)
            transform[0] = simd_float4(R_world[0], 0)
            transform[1] = simd_float4(R_world[1], 0)
            transform[2] = simd_float4(R_world[2], 0)
            transform[3] = simd_float4(centerWorld, 1)

            detections.append(Detection3D(
                center: centerWorld,
                size: size,
                yaw: yaw,
                confidence: conf,
                worldTransform: transform,
                label: boxes2D[i].label
            ))
        }

        return detections
    }

    // MARK: - Diagnostics

    /// 입력/출력 텐서의 통계를 한 줄로 찍는다 (NaN/Inf/min/max/mean).
    /// 입력 자체가 깨졌는지 vs 모델 출력이 깨졌는지 구분하기 위해.
    private static func logTensorStats(_ name: String, _ values: [Float]) {
        var nan = 0, inf = 0
        var vmin = Float.infinity, vmax = -Float.infinity, sum: Double = 0, validCount = 0
        for v in values {
            if v.isNaN { nan += 1; continue }
            if v.isInfinite { inf += 1; continue }
            if v < vmin { vmin = v }
            if v > vmax { vmax = v }
            sum += Double(v)
            validCount += 1
        }
        let mean = validCount > 0 ? Float(sum / Double(validCount)) : .nan
        if validCount == 0 { vmin = .nan; vmax = .nan }
        print(String(format: "[boxer] in.%@  N=%d  min=%.3f max=%.3f mean=%.3f  NaN=%d Inf=%d",
                     name as NSString, values.count, vmin, vmax, mean, nan, inf))
    }

    // MARK: - CoreML Inference

    private func runInference(
        image: [Float],
        sdpPatches: [Float],
        bb2d: [Float],
        rayEncoding: [Float],
        numBoxes: Int
    ) throws -> (centers: [Float], sizes: [Float], yaws: [Float], confidences: [Float]) {
        let S = Self.imageSize
        let gH = Self.gridH
        let gW = Self.gridW
        let N = Self.numPatches

        // 입력 MLMultiArray 생성. CoreML 모델은 변환 시점에 입력 dtype을 fp32로 두고
        // 가중치/연산은 fp16으로 가는 형태. fp32 입력으로 그대로 전달하면 된다.
        let imageArr = try Self.makeMultiArray(values: image,
                                               shape: [1, 3, S, S])
        let sdpArr   = try Self.makeMultiArray(values: sdpPatches,
                                               shape: [1, 1, gH, gW])
        let bb2dArr  = try Self.makeMultiArray(values: bb2d,
                                               shape: [1, numBoxes, 4])
        let rayArr   = try Self.makeMultiArray(values: rayEncoding,
                                               shape: [1, N, 6])

        let inputs: [String: MLFeatureValue] = [
            "image":        MLFeatureValue(multiArray: imageArr),
            "sdp_patches":  MLFeatureValue(multiArray: sdpArr),
            "bb2d":         MLFeatureValue(multiArray: bb2dArr),
            "ray_encoding": MLFeatureValue(multiArray: rayArr),
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)

        let outProvider = try model.prediction(from: provider)

        let centers      = try Self.extractFloats(outProvider, name: "center")
        let sizes        = try Self.extractFloats(outProvider, name: "size")
        let yaws         = try Self.extractFloats(outProvider, name: "yaw")
        let confidences  = try Self.extractFloats(outProvider, name: "confidence")

        return (centers, sizes, yaws, confidences)
    }

    /// `[Float]` → `MLMultiArray`(.float32). 데이터를 메모리 카피로 채운다.
    /// shape의 곱과 values.count 가 일치해야 한다.
    private static func makeMultiArray(values: [Float], shape: [Int]) throws -> MLMultiArray {
        let nsShape = shape.map { NSNumber(value: $0) }
        let arr = try MLMultiArray(shape: nsShape, dataType: .float32)
        let count = shape.reduce(1, *)
        precondition(values.count == count,
                     "MLMultiArray shape \(shape) (=\(count)) != values.count \(values.count)")
        // Float32 storage. UnsafeMutablePointer<Float> 로 직접 카피하는 게 가장 빠름.
        let dst = arr.dataPointer.assumingMemoryBound(to: Float.self)
        values.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: count)
        }
        return arr
    }

    /// 출력 feature 를 `[Float]` 로 추출. CoreML이 fp16/fp32/double 중 어느 dtype으로
    /// 반환하든 모두 Float 배열로 normalize 한다.
    private static func extractFloats(_ provider: MLFeatureProvider, name: String) throws -> [Float] {
        guard let value = provider.featureValue(for: name) else {
            throw NSError(domain: "BoxerNet", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Output \(name) missing from CoreML prediction"
            ])
        }
        guard let arr = value.multiArrayValue else {
            throw NSError(domain: "BoxerNet", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Output \(name) is not an MLMultiArray (type=\(value.type))"
            ])
        }
        let count = arr.count
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.assumingMemoryBound(to: Float.self)
            return Array(UnsafeBufferPointer(start: p, count: count))
        case .float16:
            // CoreML이 fp16으로 반환하면 직접 Float로 변환.
            // Swift는 Float16을 iOS 14+에서 지원.
            if #available(iOS 14.0, *) {
                let p = arr.dataPointer.assumingMemoryBound(to: Float16.self)
                return (0..<count).map { Float(p[$0]) }
            } else {
                throw NSError(domain: "BoxerNet", code: 3, userInfo: [
                    NSLocalizedDescriptionKey: "Float16 output requires iOS 14+"
                ])
            }
        case .double:
            let p = arr.dataPointer.assumingMemoryBound(to: Double.self)
            return (0..<count).map { Float(p[$0]) }
        case .int32:
            let p = arr.dataPointer.assumingMemoryBound(to: Int32.self)
            return (0..<count).map { Float(p[$0]) }
        @unknown default:
            throw NSError(domain: "BoxerNet", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Unsupported MLMultiArray dataType: \(arr.dataType.rawValue)"
            ])
        }
    }

    // MARK: - Preprocessing: SDP Patches

    /// Project LiDAR depth to 960x960 image, compute median depth per 16x16 patch.
    private func buildSDPPatches(
        depthMap: [[Float]],
        fx: Float, fy: Float, cx: Float, cy: Float
    ) -> [Float] {
        let S = Self.imageSize
        let P = Self.patchSize
        let gH = Self.gridH
        let gW = Self.gridW

        // Accumulate depths per patch.
        var patchDepths = [[Float]](repeating: [], count: gH * gW)

        let depthH = depthMap.count
        guard depthH > 0 else {
            return [Float](repeating: -1.0, count: gH * gW)
        }
        let depthW = depthMap[0].count

        // Scale factors from depth map to 960x960.
        let scaleX = Float(S) / Float(depthW)
        let scaleY = Float(S) / Float(depthH)

        // For each depth pixel, project to 960x960 and assign to a patch.
        let step = max(1, Int(sqrt(Float(depthH * depthW) / 20000.0)))
        for v in stride(from: 0, to: depthH, by: step) {
            for u in stride(from: 0, to: depthW, by: step) {
                let z = depthMap[v][u]
                guard z > 0 else { continue }

                // Pixel in 960x960 space.
                let px = Float(u) * scaleX
                let py = Float(v) * scaleY

                let pi = Int(py) / P
                let pj = Int(px) / P
                guard pi >= 0, pi < gH, pj >= 0, pj < gW else { continue }

                patchDepths[pi * gW + pj].append(z)
            }
        }

        // Compute median per patch.
        var result = [Float](repeating: -1.0, count: gH * gW)
        for idx in 0..<(gH * gW) {
            var depths = patchDepths[idx]
            guard !depths.isEmpty else { continue }
            depths.sort()
            result[idx] = depths[depths.count / 2]
        }
        return result
    }

    // MARK: - Preprocessing: Plucker Ray Encoding

    /// Compute 6D Plucker ray encoding for each patch centre.
    private func buildRayEncoding(
        T_vc: simd_float4x4,
        fx: Float, fy: Float, cx: Float, cy: Float
    ) -> [Float] {
        let P = Float(Self.patchSize)
        let gH = Self.gridH
        let gW = Self.gridW

        let R_vc = upperLeft3x3(T_vc)
        let originCam = simd_float3(0, 0, 0)
        let originVoxel = (T_vc * simd_float4(originCam, 1.0)).xyz

        var result = [Float](repeating: 0, count: gH * gW * 6)

        for i in 0..<gH {
            for j in 0..<gW {
                let u = Float(j) * P + P / 2.0
                let v = Float(i) * P + P / 2.0

                // Unproject to camera frame (pinhole).
                var dirCam = simd_float3(
                    (u - cx) / fx,
                    (v - cy) / fy,
                    1.0
                )
                dirCam = simd_normalize(dirCam)

                // Rotate to voxel frame.
                var dirVoxel = R_vc * dirCam
                dirVoxel = simd_normalize(dirVoxel)

                // Moment: m = origin x direction.
                let moment = simd_cross(originVoxel, dirVoxel)

                let idx = (i * gW + j) * 6
                result[idx + 0] = dirVoxel.x
                result[idx + 1] = dirVoxel.y
                result[idx + 2] = dirVoxel.z
                result[idx + 3] = moment.x
                result[idx + 4] = moment.y
                result[idx + 5] = moment.z
            }
        }
        return result
    }

    // MARK: - Preprocessing: Gravity Alignment

    /// Compute gravity-aligned voxel frame matching Python's
    /// `gravity_align_T_world_cam(T_wc, z_grav=True)`.
    ///
    /// - Parameter gravity_w: Gravity direction in world frame.
    ///   ARKit uses (0, -1, 0).  The original Aria VIO uses (0, 0, -1).
    private func gravityAlign(
        T_worldCam: simd_float4x4,
        gravity_w: simd_float3 = simd_float3(0, -1, 0)
    ) -> simd_float4x4 {
        let R_wc = upperLeft3x3(T_worldCam)
        let t_wc = simd_float3(T_worldCam[3].x, T_worldCam[3].y, T_worldCam[3].z)

        let g_w = simd_normalize(gravity_w)

        // Camera forward (col 2 of R_wc) in world, projected orthogonal to gravity.
        let camZ_w = R_wc * simd_float3(0, 0, 1)
        var d3 = camZ_w - g_w * simd_dot(camZ_w, g_w) // reject gravity component
        if simd_length(d3) < 1e-6 {
            d3 = d3 + simd_float3(0, 0.001, 0) // tiny offset to avoid degenerate cross
        }

        let d2 = simd_cross(d3, g_w)

        // R_world_cg: columns are [g_w, d2, d3] (gravity is X axis — Aria convention)
        var R_wcg = simd_float3x3(columns: (g_w, d2, d3))
        // Normalize columns
        R_wcg[0] = simd_normalize(R_wcg[0])
        R_wcg[1] = simd_normalize(R_wcg[1])
        R_wcg[2] = simd_normalize(R_wcg[2])

        // Extra rotation to make Z the gravity axis (z_grav=True).
        // Python: R_cg_cgz = [[0,-1,0],[0,0,1],[-1,0,0]]
        // R_world_cgz = R_world_cg @ R_cg_cgz.inverse()
        let R_cg_cgz = simd_float3x3(columns: (
            simd_float3( 0,  0, -1),
            simd_float3(-1,  0,  0),
            simd_float3( 0,  1,  0)
        ))
        let R_world_cgz = R_wcg * R_cg_cgz.inverse

        // Build 4x4.
        var T_wv = simd_float4x4(1.0)
        T_wv[0] = simd_float4(R_world_cgz[0], 0)
        T_wv[1] = simd_float4(R_world_cgz[1], 0)
        T_wv[2] = simd_float4(R_world_cgz[2], 0)
        T_wv[3] = simd_float4(t_wc, 1)
        return T_wv
    }
}

// MARK: - simd Helpers

private func upperLeft3x3(_ m: simd_float4x4) -> simd_float3x3 {
    return simd_float3x3(
        simd_float3(m[0].x, m[0].y, m[0].z),
        simd_float3(m[1].x, m[1].y, m[1].z),
        simd_float3(m[2].x, m[2].y, m[2].z)
    )
}

private func rotationZ(angle: Float) -> simd_float3x3 {
    let c = cos(angle)
    let s = sin(angle)
    return simd_float3x3(
        simd_float3(c, s, 0),
        simd_float3(-s, c, 0),
        simd_float3(0, 0, 1)
    )
}

private extension simd_float4 {
    var xyz: simd_float3 { simd_float3(x, y, z) }
}
