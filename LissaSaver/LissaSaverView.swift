// 	
//  LissaSaverView.swift
//  Lissa Saver | Three-body Orbs + Lissa Saver + Bubble Universe
//  Copyright © 2025 John Roland Penner, Toronto Island
// 	
//  Original Swift Code based on methods obtained from BASIC Code: 
//  BUBBLE/BAS using Clifford Pickover Fractal Algorithm by SCHRAF (1984)
//  LISSAJOUS/BAS from Creative Computing Magazine, and 
//  written for IBM 370 BASIC by Larry Ruane (1974)
//	
//  Swift Screen Saver for MacOS Created by John Roland Penner on 2025-10-17
//  Lissajous integrated with SyberianScene on November 18, 2025
//	Bubble Universe integrated with Syberians on November 22, 2025
//  Metal Acceleration Added November 25, 2025
//  Updated for BREW on October 27, 2025
// 

import ScreenSaver
import AVFoundation
import Metal
import MetalKit

@objc final class LissaSaverView: ScreenSaverView {

    // MARK: - Shared Syberian Three-Body Simulation State
    var syberian1Pos: CGPoint = .zero
    var syberian2Pos: CGPoint = .zero
    var syberian3Pos: CGPoint = .zero
    
    var syberianVelocities: [CGVector] = [.zero, .zero, .zero]
    var syberianAccelerations: [CGVector] = [.zero, .zero, .zero]
    
    let gravitationalConstant: CGFloat = 5000.0
    let syberianMass: CGFloat = 10.0
    let maxVelocity: CGFloat = 120.0
    let repulsionDistance: CGFloat = 100.0
    let repulsionStrength: CGFloat = 15000.0
    let dampingFactor: CGFloat = 0.995
    let syberianSize: CGFloat = 35.0
	
    private enum EffectMode: Int {
        case lissa = 0
        case bubble = 1
    }
    
    private var currentEffect: EffectMode = .lissa
    private var useGPUAcceleration: Bool = true
    
    private let defaults = UserDefaults(suiteName: "ca.theidoctor.LissaSaver")!
    private var configSheet: NSWindow?

    // MARK: - Lissa Saver State
    var lissajousPos: CGPoint = .zero
    var lissajousPhase: Float = 0.0
    var lissajousXFreq: Float = 5.0
    var lissajousYFreq: Float = 3.0
    var lissajousStepCounter: Int = 0
    var lissajousAnimTimer: TimeInterval = 0.0
    var lissajousTexture: CGImage?
    let lissajousSize: CGFloat = 120.0
    var currentLissajousScale: CGFloat = 256.0
    let lissajousScaleLerpFactor: CGFloat = 0.02
    
    let freqSteps: [(xFreq: Float, yFreq: Float)] = [
        (3.0, 2.0),
        (5.0, 4.0),
        (5.0, 3.0)
    ]

    // MARK: - Bubble Universe State
    var bubblePos: CGPoint = .zero
    var bubbleAnimTimer: TimeInterval = 0.0
    var bubbleTexture: CGImage?
    let bubbleSize: CGFloat = 500.0
    var currentBubbleScale: CGFloat = 500.0
    let bubbleScaleLerpFactor: CGFloat = 0.02
    
    var bubbleT: Double = 0.0
    var bubbleX: Double = 0.0
    var bubbleU: Double = 0.0
    var bubbleV: Double = 0.0
    let bubbleR: Double = 2.0 * .pi / 235.0
    let bubbleGridSize: Int = 200

    // MARK: - Shared Resources
    var syberianImage: NSImage?
    var backgroundImage: NSImage?
    
    var gameSound: Bool = true
    var sound18: AVAudioPlayer?
    var sound19: AVAudioPlayer?

    // MARK: - Metal Resources
    var metalDevice: MTLDevice?
    var metalCommandQueue: MTLCommandQueue?
    
    // Lissa Metal
    var lissaMetalPipeline: MTLComputePipelineState?
    
    // Bubble Metal
    var bubbleComputePipeline: MTLComputePipelineState?
    var bubbleRenderPipeline: MTLComputePipelineState?
    var bubblePointsBuffer: MTLBuffer?
    var bubbleColorsBuffer: MTLBuffer?

    // MARK: - Init
    override init?(frame: NSRect, isPreview: Bool) {
        super.init(frame: frame, isPreview: isPreview)
        
        animationTimeInterval = 1.0 / 60.0
        
        loadPreferences()
        setupSharedResources()
        initializePositions()
        setupMetalIfEnabled()
        generateCurrentTexture()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        loadPreferences()
    }
    
    private func loadPreferences() {
        currentEffect = EffectMode(rawValue: defaults.integer(forKey: "EffectMode")) ?? .lissa
        useGPUAcceleration = defaults.bool(forKey: "UseGPUAcceleration")
        if defaults.object(forKey: "UseGPUAcceleration") == nil {
            defaults.set(true, forKey: "UseGPUAcceleration")
            useGPUAcceleration = true
        }
    }
    
    private func setupSharedResources() {
        loadSyberianImage()
        loadBackgroundImage()
        setupSounds()
    }
    
    private func initializePositions() {
        let midX = bounds.midX
        let midY = bounds.midY
        syberian1Pos = CGPoint(x: midX + 300, y: midY + 200)
        syberian2Pos = CGPoint(x: midX + 500, y: midY - 200)
        syberian3Pos = CGPoint(x: midX - 300, y: midY - 200)
        
        lissajousPos = CGPoint(x: midX, y: midY + 100)
        bubblePos = CGPoint(x: midX, y: midY + 100)
        
        bubbleT = 4.0 * Double.random(in: 0...1)
    }

    // MARK: - Metal Setup
    private func setupMetalIfEnabled() {
        guard useGPUAcceleration,
              let device = MTLCreateSystemDefaultDevice() else {
            metalDevice = nil
            return
        }
        
        metalDevice = device
        metalCommandQueue = device.makeCommandQueue()
        
        if currentEffect == .lissa { setupLissaMetal(device: device) }
        if currentEffect == .bubble { setupBubbleMetal(device: device) }
    }
    
    private func setupLissaMetal(device: MTLDevice) {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;
        
        struct GlowPoint {
            float x;
            float y;
            float red;
            float green;
            float blue;
        };
        
        kernel void drawLissajousGlow(
            device GlowPoint *points [[buffer(0)]],
            texture2d<float, access::write> outTexture [[texture(0)]],
            uint2 gid [[thread_position_in_grid]],
            uint pointIndex [[thread_position_in_threadgroup]])
        {
            if (pointIndex >= 1) return;
            
            GlowPoint point = points[gid.x];
            int size = outTexture.get_width();
            
            int x = int((point.x + 1.0) * float(size - 1) / 2.0);
            int y = int((point.y + 1.0) * float(size - 1) / 2.0);
            
            if (x < 0 || x >= size || y < 0 || y >= size) return;
            
            outTexture.write(float4(point.red, point.green, point.blue, 0.784), uint2(x, y));
            
            const int2 adjacent[4] = {int2(0, -1), int2(0, 1), int2(-1, 0), int2(1, 0)};
            for (int i = 0; i < 4; i++) {
                int px = x + adjacent[i].x;
                int py = y + adjacent[i].y;
                if (px >= 0 && px < size && py >= 0 && py < size) {
                    float4 glowColor = float4(point.red * 0.55, point.green * 0.55, point.blue * 0.55, 0.471);
                    outTexture.write(glowColor, uint2(px, py));
                }
            }
        }
        """
        
        do {
            let library = try device.makeLibrary(source: shaderSource, options: nil)
            if let function = library.makeFunction(name: "drawLissajousGlow") {
                lissaMetalPipeline = try device.makeComputePipelineState(function: function)
            }
        } catch {
            print("Lissa Metal pipeline failed: \(error)")
            lissaMetalPipeline = nil
        }
    }
    
    private func setupBubbleMetal(device: MTLDevice) {
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void computeBubblePoints(
            device float2 *outPoints [[buffer(0)]],
            device float3 *outColors [[buffer(1)]],
            constant float &t [[buffer(2)]],
            constant float &r [[buffer(3)]],
            constant int &gridSize [[buffer(4)]],
            uint2 gid [[thread_position_in_grid]])
        {
            int i = gid.x;
            int j = gid.y;
            if (i >= gridSize || j >= gridSize) return;
            
            float x = 0.0;
            float u = 0.0;
            float v = 0.0;
            
            for (int iter_j = 0; iter_j <= j; iter_j++) {
                u = sin(float(i) + v) + sin(r * float(i) + x);
                v = cos(float(i) + v) + cos(r * float(i) + x);
                x = u + t;
            }
            
            int idx = j * gridSize + i;
            outPoints[idx] = float2(u, v);
            float rr = float((i * 255 / gridSize) % 256) / 255.0;
            float gg = float((j * 255 / gridSize) % 256) / 255.0;
            float bb = float(abs(255 - ((i + j) * 255 / (gridSize * 2))) % 256) / 255.0;
            outColors[idx] = float3(rr, gg, bb);
        }
        
        kernel void renderBubbleTexture(
            texture2d<float, access::write> outTexture [[texture(0)]],
            device float2 *points [[buffer(0)]],
            device float3 *colors [[buffer(1)]],
            constant int &gridSize [[buffer(2)]],
            constant int &textureSize [[buffer(3)]],
            uint idx [[thread_position_in_grid]])
        {
            if (idx >= gridSize * gridSize) return;
            
            float2 uv = points[idx];
            float3 color = colors[idx];
            
            int px = int(float(textureSize) / 2.0 + 108.0 * uv.x * float(textureSize) / 500.0);
            int py = int(float(textureSize) / 2.0 + 108.0 * uv.y * float(textureSize) / 500.0);
            
            for (int dx = 0; dx <= 1; dx++) {
                for (int dy = 0; dy <= 1; dy++) {
                    int drawX = px + dx;
                    int drawY = py + dy;
                    if (drawX >= 0 && drawX < textureSize && drawY >= 0 && drawY < textureSize) {
                        outTexture.write(float4(color, 1.0), uint2(drawX, drawY));
                    }
                }
            }
        }
        """
        
        do {
            let library = try device.makeLibrary(source: shaderSource, options: nil)
            
            guard let computeFunc = library.makeFunction(name: "computeBubblePoints"),
                  let renderFunc = library.makeFunction(name: "renderBubbleTexture") else {
                return
            }
            
            bubbleComputePipeline = try device.makeComputePipelineState(function: computeFunc)
            bubbleRenderPipeline = try device.makeComputePipelineState(function: renderFunc)
            
            let bufferSize = bubbleGridSize * bubbleGridSize
            bubblePointsBuffer = device.makeBuffer(length: bufferSize * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared)
            bubbleColorsBuffer = device.makeBuffer(length: bufferSize * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
        } catch {
            print("Bubble Metal setup failed: \(error)")
        }
    }

    // MARK: - Shared Resources Loading
    private func loadSyberianImage() {
        if let image = NSImage(named: "Syberian") {
            syberianImage = image
        }
    }
    
    private func loadBackgroundImage() {
        if let image = NSImage(named: "Hubble-NGC-3603.jpg") {
            backgroundImage = image
        }
    }
    
    private func setupSounds() {
        if let url = Bundle.main.url(forResource: "syberian", withExtension: "wav") {
            try? sound18 = AVAudioPlayer(contentsOf: url)
            sound18?.prepareToPlay()
        }
        if let url = Bundle.main.url(forResource: "syberianQix", withExtension: "wav") {
            try? sound19 = AVAudioPlayer(contentsOf: url)
            sound19?.volume = 0.40
            sound19?.prepareToPlay()
        }
    }
    
    private func playSound(_ player: AVAudioPlayer?) {
        guard gameSound, let player = player else { return }
        player.stop()
        player.currentTime = 0
        player.play()
    }

    // MARK: - Animation Loop
    override func animateOneFrame() {
        updateSyberians()
        updateCurrentEffect()
        generateCurrentTexture()
        setNeedsDisplay(bounds)
    }
    
    private func updateCurrentEffect() {
        switch currentEffect {
        case .lissa: updateLissajous()
        case .bubble: updateBubbleUniverse()
        }
    }
    
    private func generateCurrentTexture() {
        if useGPUAcceleration && metalDevice != nil {
            switch currentEffect {
            case .lissa: generateLissajousTextureMetal()
            case .bubble: generateBubbleTextureWithMetal()
            }
        } else {
            switch currentEffect {
            case .lissa: generateLissajousTextureCPU()
            case .bubble: generateBubbleTextureWithCPU()
            }
        }
    }

    // MARK: - Syberian Physics (identical to both models)
    private func updateSyberians() {
        let positions = [syberian1Pos, syberian2Pos, syberian3Pos]
        for i in 0..<3 { syberianAccelerations[i] = .zero }
        
        for i in 0..<3 {
            for j in 0..<3 where i != j {
                let dist = distance(positions[i], positions[j])
                if dist < repulsionDistance && dist > 0 {
                    let force = calculateRepulsionForce(from: positions[j], to: positions[i], distance: dist)
                    syberianAccelerations[i].dx += force.dx / syberianMass
                    syberianAccelerations[i].dy += force.dy / syberianMass
                    playSound(sound18)
                } else if dist > 0 {
                    let force = calculateGravitationalForce(from: positions[j], to: positions[i])
                    syberianAccelerations[i].dx += force.dx / syberianMass
                    syberianAccelerations[i].dy += force.dy / syberianMass
                }
            }
        }
        
        for i in 0..<3 {
            syberianVelocities[i].dx += syberianAccelerations[i].dx
            syberianVelocities[i].dy += syberianAccelerations[i].dy
            syberianVelocities[i].dx *= dampingFactor
            syberianVelocities[i].dy *= dampingFactor
            
            let speed = sqrt(syberianVelocities[i].dx * syberianVelocities[i].dx +
                           syberianVelocities[i].dy * syberianVelocities[i].dy)
            if speed > maxVelocity {
                let scale = maxVelocity / speed
                syberianVelocities[i].dx *= scale
                syberianVelocities[i].dy *= scale
            }
            
            var newX = positions[i].x + syberianVelocities[i].dx
            var newY = positions[i].y + syberianVelocities[i].dy
            let margin = syberianSize / 2
            
            if newX - margin < 0 { newX = margin; syberianVelocities[i].dx = abs(syberianVelocities[i].dx) }
            if newX + margin > bounds.width { newX = bounds.width - margin; syberianVelocities[i].dx = -abs(syberianVelocities[i].dx) }
            if newY - margin < 0 { newY = margin; syberianVelocities[i].dy = abs(syberianVelocities[i].dy) }
            if newY + margin > bounds.height { newY = bounds.height - margin; syberianVelocities[i].dy = -abs(syberianVelocities[i].dy) }
            
            switch i {
            case 0: syberian1Pos = CGPoint(x: newX, y: newY)
            case 1: syberian2Pos = CGPoint(x: newX, y: newY)
            case 2: syberian3Pos = CGPoint(x: newX, y: newY)
            default: break
            }
        }
    }
    
    private func calculateGravitationalForce(from: CGPoint, to: CGPoint) -> CGVector {
        let dx = from.x - to.x
        let dy = from.y - to.y
        let dist = sqrt(dx * dx + dy * dy)
        guard dist > 5.0 else { return .zero }
        let baseForce = gravitationalConstant * syberianMass * syberianMass / (dist * dist)
        let distanceRatio = dist / bounds.height
        let taperingFactor = 1.0 / (1.0 + distanceRatio * distanceRatio)
        let forceMagnitude = baseForce * taperingFactor
        return CGVector(dx: (dx / dist) * forceMagnitude, dy: (dy / dist) * forceMagnitude)
    }
    
    private func calculateRepulsionForce(from: CGPoint, to: CGPoint, distance: CGFloat) -> CGVector {
        let dx = to.x - from.x
        let dy = to.y - from.y
        guard distance > 0.1 else { return .zero }
        let forceMagnitude = repulsionStrength / (distance * distance)
        return CGVector(dx: (dx / distance) * forceMagnitude, dy: (dy / distance) * forceMagnitude)
    }
    
    private func distance(_ p1: CGPoint, _ p2: CGPoint) -> CGFloat {
        let dx = p2.x - p1.x
        let dy = p2.y - p1.y
        return sqrt(dx * dx + dy * dy)
    }

    // MARK: - Lissa Saver Logic
    private func updateLissajous() {
        lissajousAnimTimer += animationTimeInterval
        if lissajousAnimTimer >= 0.025 {
            lissajousAnimTimer = 0.0
            lissajousPhase += 0.01
            if lissajousPhase > 2.0 {
                lissajousPhase = 0.0
                lissajousStepCounter = (lissajousStepCounter + 1) % 3
                lissajousXFreq = freqSteps[lissajousStepCounter].xFreq
                lissajousYFreq = freqSteps[lissajousStepCounter].yFreq
                playSound(sound19)
            }
        }
        
        let centerX = (syberian1Pos.x + syberian2Pos.x + syberian3Pos.x) / 3.0
        let centerY = (syberian1Pos.y + syberian2Pos.y + syberian3Pos.y) / 3.0
        let dx = centerX - lissajousPos.x
        let dy = centerY - lissajousPos.y
        let angle = atan2(dy, dx)
        lissajousPos.x += 4.0 * cos(angle)
        lissajousPos.y += 4.0 * sin(angle)
    }
    
    private func generateLissajousTextureMetal() {
        guard let device = metalDevice,
              let queue = metalCommandQueue,
              let pipeline = lissaMetalPipeline else {
            generateLissajousTextureCPU()
            return
        }
        
        let size = Int(lissajousSize)
        let points = generateLissajousPoints()
        
        let normY1 = ((syberian1Pos.y / bounds.height) * 0.75) + 0.25
        let normY2 = ((syberian2Pos.y / bounds.height) * 0.75) + 0.25
        let normY3 = ((syberian3Pos.y / bounds.height) * 0.75) + 0.25
        let red = Float(normY1.clamped(to: 0...1))
        let green = Float(normY2.clamped(to: 0...1))
        let blue = Float(normY3.clamped(to: 0...1))
        
        struct GlowPoint { var x, y, red, green, blue: Float }
        var glowPoints = points.map { GlowPoint(x: Float($0.x), y: Float($0.y), red: red, green: green, blue: blue) }
        
        guard let pointBuffer = device.makeBuffer(bytes: &glowPoints, length: MemoryLayout<GlowPoint>.stride * glowPoints.count, options: []),
              let texture = device.makeTexture(descriptor: {
                  let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: size, height: size, mipmapped: false)
                  desc.usage = [.shaderWrite, .shaderRead]
                  return desc
              }()) else {
            generateLissajousTextureCPU()
            return
        }
        
        // Clear texture
        var clear = [UInt8](repeating: 0, count: size * size * 4)
        texture.replace(region: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0, withBytes: &clear, bytesPerRow: size * 4)
        
        guard let cmd = queue.makeCommandBuffer(),
              let encoder = cmd.makeComputeCommandEncoder() else {
            generateLissajousTextureCPU()
            return
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(pointBuffer, offset: 0, index: 0)
        encoder.setTexture(texture, index: 0)
        encoder.dispatchThreadgroups(MTLSize(width: glowPoints.count, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        encoder.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
        lissajousTexture = textureToCGImage(texture)
    }
    
    private func generateLissajousTextureCPU() {
        let size = Int(lissajousSize)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * size
        var pixelData = [UInt8](repeating: 0, count: size * size * bytesPerPixel)
        let points = generateLissajousPoints()
        
        for point in points {
            let x = Int((point.x + 1.0) * CGFloat(size - 1) / 2.0)
            let y = Int((point.y + 1.0) * CGFloat(size - 1) / 2.0)
            if x >= 0 && x < size && y >= 0 && y < size {
                drawGlowPoint(x: x, y: y, data: &pixelData, size: size)
            }
        }
        
        guard let context = CGContext(data: &pixelData, width: size, height: size, bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                     space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let cgImage = context.makeImage() else { return }
        lissajousTexture = cgImage
    }
    
    private func generateLissajousPoints() -> [CGPoint] {
        var points: [CGPoint] = []
        let pi = Float.pi
        let f1 = 2.0 * pi * lissajousXFreq
        let f2 = 2.0 * pi * lissajousYFreq
        let p2 = pi * lissajousPhase
        let f = Int(lissajousXFreq)
        guard f > 0 else { return points }
        
        let steps = 8
        for x1 in 0...steps {
            let x = Float(x1 - steps/2) / Float(steps/2)
            let t1 = arcsinApprox(x)
            let t2 = pi - t1
            
            var yValues: [Float] = []
            for i in 0..<f {
                let t3 = (t1 + 2.0 * Float(i) * pi) / f1
                let t4 = (t2 + 2.0 * Float(i) * pi) / f1
                yValues.append(sin(f2 * t3 + p2))
                yValues.append(sin(f2 * t4 + p2))
            }
            
            yValues.sort()
            var lastY: Float = -999.0
            for y in yValues {
                if abs(y - lastY) > 0.01 {
                    points.append(CGPoint(x: CGFloat(x), y: CGFloat(y)))
                    lastY = y
                }
            }
        }
        return points
    }
    
    private func arcsinApprox(_ x: Float) -> Float {
        var result = x
        if abs(x) >= 0.1 {
            result = x / (sqrt(1 + x) + sqrt(1 - x))
            result = arcsinApprox(result)
            result = 2.0 * result
        } else {
            let x2 = x * x
            let x3 = x * x2
            let x5 = x3 * x2
            let x7 = x5 * x2
            result = x + x3/6.0 + 0.075*x5 + x7/22.4
        }
        return result
    }
    
    private func drawGlowPoint(x: Int, y: Int, data: inout [UInt8], size: Int) {
        let normY1 = ((syberian1Pos.y / bounds.height) * 0.75) + 0.25
        let normY2 = ((syberian2Pos.y / bounds.height) * 0.75) + 0.25
        let normY3 = ((syberian3Pos.y / bounds.height) * 0.75) + 0.25
        let red = UInt8((normY1 * 255).clamped(to: 0...255))
        let green = UInt8((normY2 * 255).clamped(to: 0...255))
        let blue = UInt8((normY3 * 255).clamped(to: 0...255))
        
        if x >= 0 && x < size && y >= 0 && y < size {
            let o = (y * size + x) * 4
            data[o] = red; data[o+1] = green; data[o+2] = blue; data[o+3] = 200
        }
        
        let adj = [(0,-1),(0,1),(-1,0),(1,0)]
        for (dx, dy) in adj {
            let px = x + dx, py = y + dy
            if px >= 0 && px < size && py >= 0 && py < size {
                let o = (py * size + px) * 4
                data[o] = UInt8(CGFloat(red) * 0.55)
                data[o+1] = UInt8(CGFloat(green) * 0.55)
                data[o+2] = UInt8(CGFloat(blue) * 0.55)
                data[o+3] = 120
            }
        }
    }
    
    private func textureToCGImage(_ texture: MTLTexture) -> CGImage? {
        let w = texture.width, h = texture.height
        let bpr = w * 4
        var data = [UInt8](repeating: 0, count: w * h * 4)
        texture.getBytes(&data, bytesPerRow: bpr, from: MTLRegionMake2D(0,0,w,h), mipmapLevel: 0)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &data, width: w, height: h, bitsPerComponent: 8, bytesPerRow: bpr,
                                 space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        return ctx.makeImage()
    }

    // MARK: - Bubble Universe Logic
    private func updateBubbleUniverse() {
        bubbleAnimTimer += animationTimeInterval
        if bubbleAnimTimer >= 0.05 {
            bubbleAnimTimer = 0.0
            bubbleT += 0.02
            if bubbleT > 8.0 {
                bubbleT = 0.0
                playSound(sound19)
            }
        }
        
        let centerX = (syberian1Pos.x + syberian2Pos.x + syberian3Pos.x) / 3.0
        let centerY = (syberian1Pos.y + syberian2Pos.y + syberian3Pos.y) / 3.0
        let dx = centerX - bubblePos.x
        let dy = centerY - bubblePos.y
        let angle = atan2(dy, dx)
        bubblePos.x += 4.0 * cos(angle)
        bubblePos.y += 4.0 * sin(angle)
    }
    
    private func generateBubbleTextureWithMetal() {
        guard let device = metalDevice,
              let queue = metalCommandQueue,
              let compute = bubbleComputePipeline,
              let render = bubbleRenderPipeline,
              let points = bubblePointsBuffer,
              let colors = bubbleColorsBuffer else {
            generateBubbleTextureWithCPU()
            return
        }
        
        let size = Int(bubbleSize)
        guard let texture = device.makeTexture(descriptor: {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: size, height: size, mipmapped: false)
            d.usage = [.shaderWrite, .shaderRead]
            return d
        }()) else { return }
        
        guard let cmd = queue.makeCommandBuffer() else { return }
        
        // Compute pass
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(compute)
            enc.setBuffer(points, offset: 0, index: 0)
            enc.setBuffer(colors, offset: 0, index: 1)
            var t = Float(bubbleT), r = Float(bubbleR), gs = Int32(bubbleGridSize)
            enc.setBytes(&t, length: 4, index: 2)
            enc.setBytes(&r, length: 4, index: 3)
            enc.setBytes(&gs, length: 4, index: 4)
            let tg = MTLSize(width: 16, height: 16, depth: 1)
            let ng = MTLSize(width: (bubbleGridSize+15)/16, height: (bubbleGridSize+15)/16, depth: 1)
            enc.dispatchThreadgroups(ng, threadsPerThreadgroup: tg)
            enc.endEncoding()
        }
        
        // Render pass
        if let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(render)
            enc.setTexture(texture, index: 0)
            enc.setBuffer(points, offset: 0, index: 0)
            enc.setBuffer(colors, offset: 0, index: 1)
            var gs = Int32(bubbleGridSize), ts = Int32(size)
            enc.setBytes(&gs, length: 4, index: 2)
            enc.setBytes(&ts, length: 4, index: 3)
            let total = bubbleGridSize * bubbleGridSize
            let tpg = 256
            let ng = (total + tpg - 1) / tpg
            enc.dispatchThreadgroups(MTLSize(width: ng, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
            enc.endEncoding()
        }
        
        cmd.commit()
        cmd.waitUntilCompleted()
        bubbleTexture = textureToCGImage(texture)
    }
    
    private func generateBubbleTextureWithCPU() {
        let size = Int(bubbleSize)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bpp = 4
        let bpr = bpp * size
        var data = [UInt8](repeating: 0, count: size * size * bpp)
        
        bubbleX = 0.0; bubbleU = 0.0; bubbleV = 0.0
        
        for i in 0..<bubbleGridSize {
            for j in 0..<bubbleGridSize {
                bubbleU = sin(Double(i) + bubbleV) + sin(bubbleR * Double(i) + bubbleX)
                bubbleV = cos(Double(i) + bubbleV) + cos(bubbleR * Double(i) + bubbleX)
                bubbleX = bubbleU + bubbleT
                
                let px = Int(Double(size)/2 + 108.0 * bubbleU * Double(size)/500.0)
                let py = Int(Double(size)/2 + 108.0 * bubbleV * Double(size)/500.0)
                
                if px >= 0 && px < size && py >= 0 && py < size {
                    let rr = UInt8((i * 255 / bubbleGridSize) % 256)
                    let gg = UInt8((j * 255 / bubbleGridSize) % 256)
                    let bb = UInt8(abs(255 - ((i + j) * 255 / (bubbleGridSize * 2))) % 256)
                    
                    for dx in 0...1 { for dy in 0...1 {
                        let x = px + dx, y = py + dy
                        if x >= 0 && x < size && y >= 0 && y < size {
                            let o = (y * size + x) * bpp
                            data[o] = rr; data[o+1] = gg; data[o+2] = bb; data[o+3] = 255
                        }
                    }}
                }
            }
        }
        
        if let ctx = CGContext(data: &data, width: size, height: size, bitsPerComponent: 8, bytesPerRow: bpr,
                              space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
           let img = ctx.makeImage() {
            bubbleTexture = img
        }
    }

    // MARK: - Drawing
    override func draw(_ rect: NSRect) {
        NSColor.black.setFill()
        bounds.fill()
        
        backgroundImage?.draw(in: bounds, from: .zero, operation: .sourceOver, fraction: 0.25)
        
        drawSyberian(at: syberian1Pos)
        drawSyberian(at: syberian2Pos)
        drawSyberian(at: syberian3Pos)
        
        let positions = [syberian1Pos, syberian2Pos, syberian3Pos]
        let minX = positions.map(\.x).min()!, maxX = positions.map(\.x).max()!
        let minY = positions.map(\.y).min()!, maxY = positions.map(\.y).max()!
        let spread = max(maxX - minX, maxY - minY)
        
        switch currentEffect {
        case .lissa:
            let target = max((spread / 2.0) / 100.0, 2.56) * lissajousSize
            currentLissajousScale += (target - currentLissajousScale) * lissajousScaleLerpFactor
            if let tex = lissajousTexture {
                let img = NSImage(cgImage: tex, size: NSSize(width: currentLissajousScale, height: currentLissajousScale))
                let r = NSRect(x: lissajousPos.x - currentLissajousScale/2,
                              y: lissajousPos.y - currentLissajousScale/2,
                              width: currentLissajousScale, height: currentLissajousScale)
                img.draw(in: r, from: .zero, operation: .sourceOver, fraction: 0.85)
            }
            
        case .bubble:
            let target = max((spread / 4.0) / 100.0, 0.8) * bubbleSize
            currentBubbleScale += (target - currentBubbleScale) * bubbleScaleLerpFactor
            if let tex = bubbleTexture {
                let img = NSImage(cgImage: tex, size: NSSize(width: currentBubbleScale, height: currentBubbleScale))
                let r = NSRect(x: bubblePos.x - currentBubbleScale/2,
                              y: bubblePos.y - currentBubbleScale/2,
                              width: currentBubbleScale, height: currentBubbleScale)
                img.draw(in: r, from: .zero, operation: .sourceOver, fraction: 0.75)
            }
        }
    }
    
    private func drawSyberian(at position: CGPoint) {
        let rect = NSRect(x: position.x - syberianSize/2, y: position.y - syberianSize/2,
                         width: syberianSize, height: syberianSize)
        let normY = ((position.y / bounds.height) * 0.75) + 0.25
        let tint: NSColor = {
            if position == syberian1Pos { return NSColor(red: normY, green: 0, blue: 0, alpha: 0.40) }
            if position == syberian2Pos { return NSColor(red: 0, green: normY, blue: 0, alpha: 0.40) }
            return NSColor(red: 0, green: 0, blue: normY, alpha: 0.40)
        }()
        
        syberianImage?.draw(in: rect, from: .zero, operation: .sourceOver, fraction: 0.40)
        tint.setFill()
        NSBezierPath(ovalIn: rect).fill()
    }
	
	
	// MARK: - Preferences Sheet (macOS Tahoe compatible)
	override var hasConfigureSheet: Bool { true }

	override var configureSheet: NSWindow? {
		// Reuse if already created (avoids recreation)
		if let existing = configSheet { return existing }
		
		let panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 380, height: 210),
							styleMask: [.titled, .closable],
							backing: .buffered,
							defer: false)
		panel.title = "Lissa World Saver Preferences"
		panel.isReleasedWhenClosed = false		// Prevent dealloc mid-session

		let content = NSView(frame: NSRect(x: 0, y: 0, width: 380, height: 210))
		content.autoresizingMask = [.width, .height]

		// Effect selector
		let effectLabel = NSTextField(labelWithString: "Effect:")
		effectLabel.frame = NSRect(x: 20, y: 140, width: 100, height: 20)
		
		let effectPopup = NSPopUpButton(frame: NSRect(x: 120, y: 136, width: 220, height: 26))
		effectPopup.addItems(withTitles: ["Lissa Saver", "Bubble Universe"])
		effectPopup.selectItem(at: currentEffect.rawValue)
		effectPopup.target = self
		effectPopup.action = #selector(effectChanged(_:))

		// GPU checkbox
		let gpuCheckbox = NSButton(checkboxWithTitle: "Use GPU Acceleration (if available)",
								   target: self,
								   action: #selector(gpuChanged(_:)))
		gpuCheckbox.frame = NSRect(x: 20, y: 100, width: 340, height: 20)
		gpuCheckbox.state = useGPUAcceleration ? .on : .off

		// OK | Cancel buttons
		let cancelButton = NSButton(title: "Cancel", target: self, action: #selector(cancelSheet(_:)))
		//cancelButton.frame = NSRect(x: 276, y: 15, width: 84, height: 32)
		cancelButton.frame = NSRect(x: 360 - 84*2 - 20, y: 15, width: 84, height: 32)
		cancelButton.keyEquivalent = "\u{1b}"

		let okButton = NSButton(title: "OK", target: self, action: #selector(okSheet(_:)))
		//okButton.frame = NSRect(x: 360 - 84*2 - 20, y: 15, width: 84, height: 32)
		okButton.frame = NSRect(x: 276, y: 15, width: 84, height: 32)
		okButton.keyEquivalent = "\r"

		content.addSubview(effectLabel)
		content.addSubview(effectPopup)
		content.addSubview(gpuCheckbox)
		content.addSubview(cancelButton)
		content.addSubview(okButton)

		panel.contentView = content
		
		// Store the instance for dismissal
		configSheet = panel
		return panel
	}
	
	
	// MARK: - Sheet Dismissal Actions
	@objc private func okSheet(_ sender: Any) {
		if let sheet = configSheet {
			NSApp.endSheet(sheet, returnCode: NSApplication.ModalResponse.OK.rawValue)  // 1 for OK
			sheet.orderOut(nil)
			configSheet = nil
		}
	}

	@objc private func cancelSheet(_ sender: Any) {
		// Revert changes
		loadPreferences()
		setupMetalIfEnabled()
		generateCurrentTexture()
		setNeedsDisplay(bounds)
		
		if let sheet = configSheet {
			NSApp.endSheet(sheet, returnCode: NSApplication.ModalResponse.cancel.rawValue)  // 0 for Cancel
			sheet.orderOut(nil)
			configSheet = nil
		}
	}
	
	
	// MARK: - Sheet Actions
	@objc private func effectChanged(_ sender: NSPopUpButton) {
		guard let newEffect = EffectMode(rawValue: sender.indexOfSelectedItem) else { return }
		currentEffect = newEffect
		defaults.set(newEffect.rawValue, forKey: "EffectMode")
		setupMetalIfEnabled()
		generateCurrentTexture()
		setNeedsDisplay(bounds)
	}

	@objc private func gpuChanged(_ sender: NSButton) {
		useGPUAcceleration = (sender.state == .on)
		defaults.set(useGPUAcceleration, forKey: "UseGPUAcceleration")
		setupMetalIfEnabled()
		generateCurrentTexture()
		setNeedsDisplay(bounds)
	}

}

// Helper extension
extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        return min(max(self, range.lowerBound), range.upperBound)
    }
}
