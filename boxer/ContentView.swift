//
//  ContentView.swift
//  boxer
//
//  Created by Bharath Kumar Adinarayan on 09.04.26.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = ARViewModel()

    var body: some View {
        ZStack {
            ARViewContainer(viewModel: viewModel)

            VStack {
                // Status bar
                HStack {
                    Text(viewModel.status)
                        .font(.system(size: 14, weight: .medium, design: .monospaced))
                        .foregroundColor(.white)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(.black.opacity(0.6))
                        .cornerRadius(8)
                    Spacer()
                }
                .padding(.top, 50)
                .padding(.horizontal, 16)

                Spacer()

                // Bottom controls
                HStack(spacing: 20) {
                    // Detect button
                    Button(action: { viewModel.detectNow() }) {
                        HStack {
                            Image(systemName: "cube.transparent")
                            Text("Detect 3D")
                        }
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 14)
                        .background(.blue)
                        .cornerRadius(12)
                    }
                    .disabled(viewModel.isProcessing)
                    .opacity(viewModel.isProcessing ? 0.5 : 1.0)

                    // Clear button
                    Button(action: { viewModel.clearBoxes() }) {
                        HStack {
                            Image(systemName: "trash")
                            Text("Clear")
                        }
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 24)
                        .padding(.vertical, 14)
                        .background(.red.opacity(0.8))
                        .cornerRadius(12)
                    }

                    // Confidence slider
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Conf: \(viewModel.confidenceThreshold, specifier: "%.1f")")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(.white)
                        Slider(value: $viewModel.confidenceThreshold, in: 0.1...0.9, step: 0.1)
                            .frame(width: 100)
                    }
                }
                .padding(.bottom, 40)
                .padding(.horizontal, 16)
            }

            // Detection list overlay
            if !viewModel.detections.isEmpty {
                VStack {
                    Spacer()
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 8) {
                            ForEach(viewModel.detections) { det in
                                DetectionCard(detection: det)
                            }
                        }
                        .padding(.horizontal, 16)
                    }
                    .padding(.bottom, 110)
                }
            }
        }
    }
}

struct DetectionCard: View {
    let detection: DetectionInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(detection.label)
                .font(.system(size: 14, weight: .bold))
                .foregroundColor(.white)
            Text(String(format: "%.0fx%.0fx%.0f cm",
                        detection.size.x * 100,
                        detection.size.y * 100,
                        detection.size.z * 100))
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.white.opacity(0.8))
            Text(String(format: "conf: %.0f%%", detection.confidence * 100))
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.green)
        }
        .padding(10)
        .background(.black.opacity(0.7))
        .cornerRadius(8)
    }
}
