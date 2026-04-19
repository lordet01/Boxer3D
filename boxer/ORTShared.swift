import Foundation
import OnnxRuntimeBindings

/// Single ORT environment for all sessions (avoids duplicate allocators / state).
enum ORTShared {
    static let env: ORTEnv = {
        do {
            return try ORTEnv(loggingLevel: .warning)
        } catch {
            fatalError("ORTEnv failed: \(error)")
        }
    }()

    /// CoreML EP 컴파일 결과를 저장할 디렉터리. 모델별로 분리해 충돌을 피한다.
    /// `Caches`는 OS가 공간 부족 시 비울 수 있지만, 비워져도 다음 실행에서 재컴파일될 뿐
    /// 동작에는 문제가 없다(컴파일 캐시이지 모델 자체가 아님).
    static func coreMLCacheDir(subdir: String) -> String {
        let base = FileManager.default
            .urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ort_coreml", isDirectory: true)
            .appendingPathComponent(subdir, isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base.path
    }
}
