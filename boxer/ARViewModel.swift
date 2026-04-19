import Foundation
import ARKit
import Combine
import OSLog
import SceneKit
import simd
import MachO
import Darwin

/// 화면(`status`)과 Xcode 콘솔(`print` + `os.Logger`)에 동시에 기록하기 위한 로거.
/// Xcode Debug area에 `[boxer] ...` 접두어로 보입니다.
private let appLog = Logger(subsystem: "bharath.boxer", category: "ARViewModel")

/// 현재 프로세스의 phys_footprint(=Jetsam이 보는 실제 사용량)를 MB 단위로 반환.
/// detect 직전/직후에 찍어서 OS에 의한 메모리 kill 여부를 빠르게 분간할 수 있다.
@inline(__always)
private func memMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard result == KERN_SUCCESS else { return -1 }
    return Double(info.phys_footprint) / (1024.0 * 1024.0)
}

struct DetectionInfo: Identifiable {
    let id = UUID()
    let label: String
    let size: simd_float3
    let confidence: Float
}

@MainActor
final class ARViewModel: ObservableObject {
    @Published var status: String = "Initializing..."
    @Published var isProcessing: Bool = false
    @Published var detections: [DetectionInfo] = []
    /// 다중 박스를 보려면 0.3~0.5 권장. UI 슬라이더로 0.1~0.9 사이 조정 가능.
    @Published var confidenceThreshold: Float = 0.3

    /// YOLO 결과 중 BoxerNet에 넘길 후보 박스 최대 개수.
    /// CoreML/ANE 가속 위해 `BoxerNet.fixedNumBoxes`로 정수 고정. (free-dim override)
    /// 더 늘리려면 `BoxerNet.fixedNumBoxes`도 같이 키워야 한다.
    private var maxBoxesPerFrame: Int { BoxerNet.fixedNumBoxes }

    var sceneView: ARSCNView?
    private var boxerNet: BoxerNet?
    private var yoloDetector: YOLODetector?
    private var boxNodes: [SCNNode] = []

    /// BoxerNet 가 출력하는 `center` 가 박스의 어느 점인지에 대한 컨벤션.
    /// 실측 결과 BoxerNet 라벨은 **top-center** (객체 윗면 중앙) 였다.
    ///
    /// - `.topCenter` (기본/정답): 모델 center = 박스 윗면 중앙.
    ///   → 박스 중심을 voxel `+Z` 방향으로 `-size.z/2` 만큼 내린다 (= 아래로 절반).
    /// - `.centroid`: 모델 center = 박스 무게중심. lift 없음.
    /// - `.floorCenter`: 모델 center = 박스 바닥 중심. `+size.z/2` 위로.
    ///
    /// 검증 시나리오에 따라 토글하기 위해 enum으로 두지만, 운영에서는 `.topCenter` 고정.
    enum BoxAnchorMode { case topCenter, centroid, floorCenter }
    static var anchorMode: BoxAnchorMode = .topCenter

    func setup(sceneView: ARSCNView) {
        self.sceneView = sceneView
        Task.detached { await self.loadModelsInBackground() }
    }

    // MARK: - Reporting

    /// UI 라벨과 Xcode 콘솔 양쪽에 메시지를 남긴다.
    /// `error`를 넘기면 콘솔 쪽에는 원래 에러 객체가 함께 찍힌다(디버깅용 상세 포함).
    @MainActor
    private func report(_ message: String,
                        error: Error? = nil,
                        file: String = #fileID,
                        line: Int = #line) {
        self.status = message
        let tag = "[boxer] \(file):\(line)"
        if let error {
            let nsErr = error as NSError
            print("\(tag) \(message) — \(error)  [\(nsErr.domain) code=\(nsErr.code)]")
            appLog.error("\(message, privacy: .public) — \(String(describing: error), privacy: .public)")
        } else {
            print("\(tag) \(message)")
            appLog.log("\(message, privacy: .public)")
        }
    }

    /// `nonisolated` 컨텍스트(백그라운드 Task)에서 쓸 수 있는 편의 래퍼.
    private nonisolated func reportAsync(_ message: String,
                                         error: Error? = nil,
                                         file: String = #fileID,
                                         line: Int = #line) async {
        await MainActor.run { self.report(message, error: error, file: file, line: line) }
    }

    // MARK: - Model Loading

    nonisolated private func loadModelsInBackground() async {
        let yoloPath = Bundle.main.path(forResource: "yolo11n", ofType: "onnx")
        // CoreML 모델은 Xcode 빌드 시 자동으로 .mlpackage → .mlmodelc 로 컴파일된다.
        // 파일 이름을 `BoxerNetModel` 로 둔 이유: Xcode가 mlpackage 이름과 같은
        // Swift 클래스(`BoxerNetModel`)를 자동 생성하기 때문에, 우리 wrapper 클래스
        // `BoxerNet` 과의 이름 충돌을 회피하기 위함이다.
        let boxerPath = Bundle.main.path(forResource: "BoxerNetModel", ofType: "mlmodelc")
            ?? Bundle.main.path(forResource: "BoxerNetModel", ofType: "mlpackage")

        await reportAsync("Loading YOLO...")
        guard let yoloPath else {
            await reportAsync("yolo11n.onnx not found")
            return
        }
        let yolo: YOLODetector
        do { yolo = try YOLODetector(modelPath: yoloPath) }
        catch {
            await reportAsync("YOLO failed: \(error.localizedDescription)", error: error)
            return
        }

        await reportAsync("Loading BoxerNet...")
        guard let boxerPath else {
            await reportAsync("BoxerNetModel.mlpackage not found in bundle")
            return
        }
        let boxer: BoxerNet
        do { boxer = try BoxerNet(modelPath: boxerPath) }
        catch {
            await reportAsync("BoxerNet failed: \(error.localizedDescription)", error: error)
            return
        }

        await MainActor.run {
            self.yoloDetector = yolo
            self.boxerNet = boxer
            self.report("Warming up models…")
        }

        // 워밍업: 첫 detect 시점이 아니라 모델 로딩 직후 백그라운드에서 1회 추론을 돌려
        // CoreML 컴파일/캐시 생성을 미리 끝낸다. 두 번째 실행부터는 캐시 적중으로 거의 즉시.
        let t0 = CFAbsoluteTimeGetCurrent()
        print(String(format: "[boxer] mem.before-warmup = %.0f MB", memMB()))
        do {
            try yolo.warmup()
            let dt1 = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            print(String(format: "[boxer] mem.after-yolo   = %.0f MB", memMB()))
            try boxer.warmup()
            let dt2 = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            print(String(format: "[boxer] mem.after-boxer  = %.0f MB", memMB()))
            print(String(format: "[boxer] Warmup done — yolo=%.0fms, total=%.0fms", dt1, dt2))
            await reportAsync("Ready — tap Detect 3D")
        } catch {
            print("[boxer] Warmup failed: \(error)  — first detect will pay the cost")
            await reportAsync("Ready — tap Detect 3D")
        }
    }

    // MARK: - Detection

    func detectNow() {
        guard let sceneView, let frame = sceneView.session.currentFrame,
              let boxerNet, let yoloDetector else {
            report("Not ready"); return
        }
        guard frame.sceneDepth != nil else {
            report("No LiDAR depth"); return
        }

        isProcessing = true
        report("Detecting...")

        Task.detached {
            do {
                let results = try await self.runPipeline(frame: frame, boxer: boxerNet, yolo: yoloDetector)
                await MainActor.run {
                    self.placeBoxes(results, in: sceneView)
                    self.isProcessing = false
                }
            } catch {
                await self.reportAsync("Error: \(error.localizedDescription)", error: error)
                await MainActor.run { self.isProcessing = false }
            }
        }
    }

    nonisolated private func runPipeline(
        frame: ARFrame, boxer: BoxerNet, yolo: YOLODetector
    ) async throws -> [Detection3D] {
        let topK = await MainActor.run { self.maxBoxesPerFrame }
        let conf = await MainActor.run { self.confidenceThreshold }

        let tStart = CFAbsoluteTimeGetCurrent()
        print(String(format: "[boxer] mem.start         = %.0f MB", memMB()))

        // 1. 전처리.
        let (boxerImage, _, _) = pixelBufferToFloatArray(frame.capturedImage, targetSize: BoxerNet.imageSize)
        let (yoloImage, _, _) = pixelBufferToFloatArray(frame.capturedImage, targetSize: 640)
        let tPrep = CFAbsoluteTimeGetCurrent()
        print(String(format: "[boxer] mem.after-prep    = %.0f MB", memMB()))

        // 2. YOLO 2D detection.
        let yoloBoxes = try yolo.detect(image: yoloImage, imageWidth: 640, imageHeight: 640)
        let tYolo = CFAbsoluteTimeGetCurrent()
        guard !yoloBoxes.isEmpty else {
            print(String(format: "[boxer] timing  prep=%.0fms  yolo=%.0fms  no objects",
                         (tPrep - tStart) * 1000, (tYolo - tPrep) * 1000))
            await reportAsync("No objects detected")
            return []
        }

        // top-K 후보 (스코어 내림차순). top-K 늘리면 다중 객체 검출이 가능.
        let topBoxes = Array(yoloBoxes.sorted { $0.score > $1.score }.prefix(topK))
        let scale = Float(BoxerNet.imageSize) / 640.0
        let boxes2D = topBoxes.map { box in
            Box2D(xmin: box.xmin * scale, ymin: box.ymin * scale,
                  xmax: box.xmax * scale, ymax: box.ymax * scale,
                  label: box.label, score: box.score)
        }

        // 3. 깊이 + intrinsics.
        let depthMap = extractDepthMap(frame.sceneDepth!.depthMap)
        let intrinsics = scaleIntrinsicsWithCrop(
            frame.camera.intrinsics,
            from: frame.camera.imageResolution,
            toSize: BoxerNet.imageSize
        )
        let tDepth = CFAbsoluteTimeGetCurrent()

        // 4. BoxerNet 3D lifting (BoxerNet 자체는 confidenceThreshold=0으로 호출해서
        //    raw confidence 분포를 로그에 찍은 뒤, 여기서 직접 필터링한다.
        //    이렇게 해야 “왜 1개만 나오는지” 진단이 명확하다.)
        print(String(format: "[boxer] mem.before-boxer  = %.0f MB", memMB()))
        let raw = try boxer.predict(
            image: boxerImage, depthMap: depthMap, intrinsics: intrinsics,
            cameraTransform: frame.camera.transform, boxes2D: boxes2D,
            confidenceThreshold: 0.0
        )
        let tBoxer = CFAbsoluteTimeGetCurrent()
        print(String(format: "[boxer] mem.after-boxer   = %.0f MB", memMB()))

        let kept = raw.filter { $0.confidence >= conf }
        let confs = raw.map { $0.confidence }.sorted(by: >)

        print(String(format:
            "[boxer] timing  prep=%.0fms  yolo=%.0fms  depth=%.0fms  boxer=%.0fms  total=%.0fms",
            (tPrep  - tStart) * 1000,
            (tYolo  - tPrep)  * 1000,
            (tDepth - tYolo)  * 1000,
            (tBoxer - tDepth) * 1000,
            (tBoxer - tStart) * 1000))
        let confStr = confs.prefix(8).map { String(format: "%.2f", $0) }.joined(separator: ", ")
        print("[boxer] yolo=\(yoloBoxes.count)  -> top-K=\(boxes2D.count)  -> boxer=\(raw.count)  conf=[\(confStr)]  threshold=\(conf)  -> kept=\(kept.count)")

        await reportAsync(kept.isEmpty
                          ? "0/\(raw.count) above conf \(String(format: "%.2f", conf)) (slider ↓?)"
                          : "Ready — \(kept.count)/\(raw.count) boxes")
        return kept
    }

    // MARK: - 3D Box Rendering

    private func placeBoxes(_ detections: [Detection3D], in sceneView: ARSCNView) {
        clearBoxes()

        let colors: [UIColor] = [.systemRed, .systemGreen, .systemBlue]

        // ARKit world Y 가 up. 카메라 / floor 위치 기준으로 박스가 어디 있는지 한번에 본다.
        let camY: Float = sceneView.session.currentFrame?.camera.transform.columns.3.y ?? .nan

        for (i, det) in detections.enumerated() {
            let color = colors[i % colors.count]

            // Semi-transparent fill.
            let box = SCNBox(width: CGFloat(det.size.x), height: CGFloat(det.size.y),
                             length: CGFloat(det.size.z), chamferRadius: 0)
            let mat = SCNMaterial()
            mat.diffuse.contents = color.withAlphaComponent(0.3)
            mat.isDoubleSided = true
            box.materials = [mat]

            // 박스 중심을 voxel +Z(=world up) 방향으로 anchorMode 에 맞춰 보정한다.
            // 모델 center 가 top  → 박스 중심을 -size.z/2 (아래로 절반)
            //          centroid  → 보정 없음
            //          floor    → +size.z/2 (위로 절반)
            // 박스의 voxel +Z 축 = worldTransform.columns.2.
            var transform = det.worldTransform
            let upInWorldRaw = simd_normalize(simd_float3(
                transform.columns.2.x,
                transform.columns.2.y,
                transform.columns.2.z
            ))
            let lift: Float
            switch Self.anchorMode {
            case .topCenter:   lift = -det.size.z / 2
            case .centroid:    lift = 0
            case .floorCenter: lift = +det.size.z / 2
            }
            if lift != 0 {
                let shifted = det.center + upInWorldRaw * lift
                transform.columns.3 = simd_float4(shifted, 1)
            }

            // 박스의 "위/아래" 길이는 voxel +Z 방향으로 펼쳐진다.
            // (size.z 가 height — gravityAlign() 결과 voxel Z = world up.)
            let upInWorld = simd_normalize(simd_float3(
                transform.columns.2.x,
                transform.columns.2.y,
                transform.columns.2.z
            ))
            let centerWorld = simd_float3(
                transform.columns.3.x,
                transform.columns.3.y,
                transform.columns.3.z
            )
            // box bottom / top in world = center ± up * size.z/2.
            let bottomWorld = centerWorld - upInWorld * (det.size.z / 2)
            let topWorld    = centerWorld + upInWorld * (det.size.z / 2)
            // 카메라 대비, world Y(=up) 좌표:
            //   - centerY  : 박스 중심의 절대 높이
            //   - bottomY  : 박스 하단의 절대 높이 (=객체가 놓인 면이면 floor 와 같아야 함)
            //   - up.y     : voxel up 이 world up 과 얼마나 정렬됐는지 (1.0 이상적)
            //   - vsCamY   : 카메라 높이 대비 박스 중심의 상대 높이 (cm)
            print(String(format:
                "[boxer] place#%d  size=(%.2f,%.2f,%.2f)m  " +
                "center=(%.2f,%.2f,%.2f)  bottomY=%.2f  topY=%.2f  " +
                "up.y=%.3f  cam.y=%.2f  centerVsCam=%+.0fcm  mode=%@",
                i,
                det.size.x, det.size.y, det.size.z,
                centerWorld.x, centerWorld.y, centerWorld.z,
                bottomWorld.y, topWorld.y,
                upInWorld.y, camY,
                (centerWorld.y - camY) * 100,
                {
                    switch Self.anchorMode {
                    case .topCenter:   return "top"
                    case .centroid:    return "centroid"
                    case .floorCenter: return "floor"
                    }
                }() as String
            ))

            let node = SCNNode(geometry: box)
            node.simdWorldTransform = transform

            // Thick wireframe edges (12 cylinders).
            addWireframe(to: node, size: det.size, color: color, radius: 0.003)

            // Floating label.
            let label = det.label ?? "object"
            let sizeStr = String(format: "%.0fx%.0fx%.0f cm",
                                 det.size.x * 100, det.size.y * 100, det.size.z * 100)
            addLabel("\(label)\n\(sizeStr)", to: node, offset: det.size.y / 2 + 0.03)

            sceneView.scene.rootNode.addChildNode(node)
            boxNodes.append(node)
        }

        detections.forEach { det in
            self.detections.append(DetectionInfo(
                label: det.label ?? "object", size: det.size, confidence: det.confidence
            ))
        }
    }

    private func addWireframe(to parent: SCNNode, size: simd_float3, color: UIColor, radius: Float) {
        let hw = size.x / 2, hh = size.y / 2, hd = size.z / 2
        let edgeMat = SCNMaterial()
        edgeMat.diffuse.contents = color

        let edges: [(simd_float3, simd_float3)] = [
            (simd_float3(-hw, -hh, -hd), simd_float3( hw, -hh, -hd)),
            (simd_float3( hw, -hh, -hd), simd_float3( hw, -hh,  hd)),
            (simd_float3( hw, -hh,  hd), simd_float3(-hw, -hh,  hd)),
            (simd_float3(-hw, -hh,  hd), simd_float3(-hw, -hh, -hd)),
            (simd_float3(-hw,  hh, -hd), simd_float3( hw,  hh, -hd)),
            (simd_float3( hw,  hh, -hd), simd_float3( hw,  hh,  hd)),
            (simd_float3( hw,  hh,  hd), simd_float3(-hw,  hh,  hd)),
            (simd_float3(-hw,  hh,  hd), simd_float3(-hw,  hh, -hd)),
            (simd_float3(-hw, -hh, -hd), simd_float3(-hw,  hh, -hd)),
            (simd_float3( hw, -hh, -hd), simd_float3( hw,  hh, -hd)),
            (simd_float3( hw, -hh,  hd), simd_float3( hw,  hh,  hd)),
            (simd_float3(-hw, -hh,  hd), simd_float3(-hw,  hh,  hd)),
        ]

        for (a, b) in edges {
            let cyl = SCNCylinder(radius: CGFloat(radius), height: CGFloat(simd_distance(a, b)))
            cyl.materials = [edgeMat]
            let node = SCNNode(geometry: cyl)
            node.simdPosition = (a + b) / 2
            let dir = simd_normalize(b - a)
            let dot = simd_dot(simd_float3(0, 1, 0), dir)
            if abs(dot) < 0.999 {
                let axis = simd_normalize(simd_cross(simd_float3(0, 1, 0), dir))
                node.simdRotation = simd_float4(axis, acos(dot))
            }
            parent.addChildNode(node)
        }
    }

    private func addLabel(_ text: String, to parent: SCNNode, offset: Float) {
        let scnText = SCNText(string: text, extrusionDepth: 0.005)
        scnText.font = UIFont.systemFont(ofSize: 0.03, weight: .bold)
        scnText.firstMaterial?.diffuse.contents = UIColor.white
        scnText.flatness = 0.1
        let node = SCNNode(geometry: scnText)
        node.position = SCNVector3(-0.05, offset, 0)
        node.constraints = [SCNBillboardConstraint()]
        parent.addChildNode(node)
    }

    func clearBoxes() {
        boxNodes.forEach { $0.removeFromParentNode() }
        boxNodes.removeAll()
        detections.removeAll()
    }
}

// MARK: - Image Helpers

func pixelBufferToFloatArray(
    _ pixelBuffer: CVPixelBuffer,
    targetSize: Int = BoxerNet.imageSize
) -> ([Float], Int, Int) {
    var ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let context = CIContext()

    // Center-crop to square.
    let w = ciImage.extent.width, h = ciImage.extent.height
    let side = min(w, h)
    ciImage = ciImage.cropped(to: CGRect(x: (w - side) / 2, y: (h - side) / 2,
                                          width: side, height: side))

    // Resize to target.
    let scale = CGFloat(targetSize) / side
    let resized = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

    // Render to RGBA.
    var rgba = [UInt8](repeating: 0, count: targetSize * targetSize * 4)
    context.render(resized, toBitmap: &rgba, rowBytes: targetSize * 4,
                   bounds: CGRect(x: resized.extent.origin.x, y: resized.extent.origin.y,
                                  width: CGFloat(targetSize), height: CGFloat(targetSize)),
                   format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())

    // RGBA → CHW float32.
    let n = targetSize * targetSize
    var result = [Float](repeating: 0, count: 3 * n)
    for i in 0..<n {
        result[i]         = Float(rgba[i * 4])     / 255.0
        result[n + i]     = Float(rgba[i * 4 + 1]) / 255.0
        result[2 * n + i] = Float(rgba[i * 4 + 2]) / 255.0
    }
    return (result, targetSize, targetSize)
}

func extractDepthMap(_ depthBuffer: CVPixelBuffer) -> [[Float]] {
    CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly) }

    let h = CVPixelBufferGetHeight(depthBuffer)
    let w = CVPixelBufferGetWidth(depthBuffer)
    let bytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)
    let base = CVPixelBufferGetBaseAddress(depthBuffer)!

    var result = [[Float]](repeating: [Float](repeating: 0, count: w), count: h)
    for y in 0..<h {
        let row = base.advanced(by: y * bytesPerRow).assumingMemoryBound(to: Float32.self)
        for x in 0..<w { result[y][x] = row[x] }
    }
    return result
}

func scaleIntrinsicsWithCrop(
    _ intrinsics: simd_float3x3, from: CGSize, toSize: Int
) -> simd_float3x3 {
    let w = Float(from.width), h = Float(from.height)
    let side = min(w, h)
    let scale = Float(toSize) / side

    var s = intrinsics
    s[0][0] *= scale                                     // fx
    s[1][1] *= scale                                     // fy
    s[2][0] = (intrinsics[2][0] - (w - side) / 2) * scale  // cx
    s[2][1] = (intrinsics[2][1] - (h - side) / 2) * scale  // cy
    return s
}
