import SwiftUI
import ARKit
import SceneKit

struct ARViewContainer: UIViewRepresentable {
    @ObservedObject var viewModel: ARViewModel

    func makeUIView(context: Context) -> ARSCNView {
        let sceneView = ARSCNView()
        sceneView.autoenablesDefaultLighting = true
        sceneView.automaticallyUpdatesLighting = true

        // Configure AR session with LiDAR.
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth]
        config.planeDetection = [.horizontal, .vertical]

        sceneView.session.run(config)
        viewModel.setup(sceneView: sceneView)

        return sceneView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}
