import UIKit
import MetalKit
import QuartzCore
import ComponentFlow

class LiquidGlassV2MetalLayer: CAMetalLayer {

    private var commandQueue: MTLCommandQueue?
    private var pipelineState: MTLRenderPipelineState?
    private var backgroundTexture: MTLTexture?
    private var vertexBuffer: MTLBuffer?

    private var currentUVOffset: SIMD2<Float> = .zero
    private var currentUVScale: SIMD2<Float> = SIMD2<Float>(1, 1)
    private var lastImageSize: CGSize = .zero

    var blurRadius: CGFloat = 0
    var curvature: CGFloat = 0.3
    var domeHeight: CGFloat = 0.02
    var maxEdgeBlur: CGFloat = 1.5
    var lightAngle: CGFloat = .pi / 2
    var lightIntensity: CGFloat = 0.5
    var ambientStrength: CGFloat = 0.3
    var overlayTint: UIColor = UIColor(white: 1.0, alpha: 0.65)
    var saturation: CGFloat = 1.6
    var chromaticAberration: CGFloat = 0.3
    var thickness: CGFloat = 0.03
    var frostIntensity: CGFloat = 0.8
    var fresnelIntensity: CGFloat = 0.3
    var innerShadowIntensity: CGFloat = 0.15
    var morphScale: CGSize = CGSize(width: 1.0, height: 1.0)

    var backgroundContentProvider: (() -> (UIImage?, CGRect)?)?
    var shapeCornerRadius: CGFloat = 0

    override init() {
        super.init()
        commonInit()
    }

    override init(layer: Any) {
        super.init(layer: layer)
        if let sourceLayer = layer as? LiquidGlassV2MetalLayer {
            self.commandQueue = sourceLayer.commandQueue
            self.pipelineState = sourceLayer.pipelineState
            self.backgroundTexture = sourceLayer.backgroundTexture
            self.vertexBuffer = sourceLayer.vertexBuffer
            self.currentUVOffset = sourceLayer.currentUVOffset
            self.currentUVScale = sourceLayer.currentUVScale
            self.lastImageSize = sourceLayer.lastImageSize
            self.backgroundContentProvider = sourceLayer.backgroundContentProvider
            self.shapeCornerRadius = sourceLayer.shapeCornerRadius
            self.blurRadius = sourceLayer.blurRadius
            self.lightAngle = sourceLayer.lightAngle
            self.lightIntensity = sourceLayer.lightIntensity
            self.ambientStrength = sourceLayer.ambientStrength
            self.overlayTint = sourceLayer.overlayTint
            self.saturation = sourceLayer.saturation
            self.chromaticAberration = sourceLayer.chromaticAberration
            self.thickness = sourceLayer.thickness
            self.frostIntensity = sourceLayer.frostIntensity
            self.fresnelIntensity = sourceLayer.fresnelIntensity
            self.innerShadowIntensity = sourceLayer.innerShadowIntensity
            self.morphScale = sourceLayer.morphScale
        }
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        commonInit()
    }

    private func commonInit() {
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }

        self.device = metalDevice
        self.pixelFormat = .bgra8Unorm
        self.framebufferOnly = true
        self.isOpaque = false
        self.backgroundColor = UIColor.clear.cgColor
        self.contentsScale = UIScreen.main.scale
        self.needsDisplayOnBoundsChange = true
        self.presentsWithTransaction = true

        setupMetal()
    }

    private func setupMetal() {
        guard let device = device else { return }

        commandQueue = device.makeCommandQueue()

        let vertices: [Float] = [
            -1.0, -1.0,  0.0, 1.0,
             1.0, -1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 0.0,
             1.0,  1.0,  1.0, 0.0
        ]

        vertexBuffer = device.makeBuffer(bytes: vertices,
                                         length: vertices.count * MemoryLayout<Float>.stride,
                                         options: [])

        let library: MTLLibrary?
        #if os(iOS)
        let mainBundle = Bundle(for: LiquidGlassV2View.self)
        if let path = mainBundle.path(forResource: "GlassBackgroundComponentMetalSourcesBundle", ofType: "bundle"),
           let bundle = Bundle(path: path) {
            library = try? device.makeDefaultLibrary(bundle: bundle)
        } else {
            library = device.makeDefaultLibrary()
        }
        #else
        library = device.makeDefaultLibrary()
        #endif

        guard let library = library else {
            print("Failed to create Metal library")
            return
        }

        let vertexFunction = library.makeFunction(name: "liquidGlassV2Vertex")
        let fragmentFunction = library.makeFunction(name: "liquidGlassV2Fragment")

        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float2
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0

        vertexDescriptor.attributes[1].format = .float2
        vertexDescriptor.attributes[1].offset = MemoryLayout<Float>.stride * 2
        vertexDescriptor.attributes[1].bufferIndex = 0

        vertexDescriptor.layouts[0].stride = MemoryLayout<Float>.stride * 4
        vertexDescriptor.layouts[0].stepRate = 1
        vertexDescriptor.layouts[0].stepFunction = .perVertex

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm

        pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true
        pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
        pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha

        pipelineDescriptor.vertexDescriptor = vertexDescriptor

        do {
            pipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            print("Error creating pipeline state: \(error)")
        }
    }

    private func updateBackgroundTexture() {
        guard let device = device else { return }

        let content = backgroundContentProvider?()
        let cgImage = content?.0?.cgImage
        guard let rect = content?.1 else { return }

        lastImageSize = cgImage.map { CGSize(width: $0.width, height: $0.height) } ?? lastImageSize

        let scale = UIScreen.main.scale
        let imageWidth = CGFloat(lastImageSize.width)
        let imageHeight = CGFloat(lastImageSize.height)

        currentUVOffset = SIMD2<Float>(
            Float(rect.origin.x * scale / imageWidth),
            Float(rect.origin.y * scale / imageHeight)
        )
        currentUVScale = SIMD2<Float>(
            Float(rect.width * scale / imageWidth),
            Float(rect.height * scale / imageHeight)
        )

        guard let cgImage else { return }

        let textureLoader = MTKTextureLoader(device: device)

        do {
            let texture = try textureLoader.newTexture(cgImage: cgImage, options: [
                .textureUsage: NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
                .textureStorageMode: NSNumber(value: MTLStorageMode.shared.rawValue),
                .SRGB: false
            ])

            backgroundTexture = texture
        } catch {
            print("Error creating texture: \(error)")
        }
    }

    override func display() {
        guard (presentation() ?? self).opacity != 0, let drawable = nextDrawable() else { return }

        updateBackgroundTexture()

        guard
              let pipelineState = pipelineState,
              let commandQueue = commandQueue,
              let backgroundTexture = backgroundTexture
        else {
            return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }

        let renderPassDescriptor = MTLRenderPassDescriptor()
        renderPassDescriptor.colorAttachments[0].texture = drawable.texture
        renderPassDescriptor.colorAttachments[0].loadAction = .clear
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        renderPassDescriptor.colorAttachments[0].storeAction = .store

        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }

        renderEncoder.setRenderPipelineState(pipelineState)
        renderEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderEncoder.setFragmentTexture(backgroundTexture, index: 0)

        let width = Float(round(bounds.width))
        let height = Float(round(bounds.height))
        let aspectRatio = height / width

        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        overlayTint.getRed(&r, green: &g, blue: &b, alpha: &a)

        var params = LiquidGlassV2Params(
            radius: Float(shapeCornerRadius) / width,
            aspectRatio: aspectRatio,
            uvOffset: currentUVOffset,
            uvScale: currentUVScale,
            p: Float(curvature),
            height: Float(domeHeight),
            maxEdgeBlur: Float(maxEdgeBlur),
            lightAngle: Float(lightAngle),
            lightIntensity: Float(lightIntensity),
            ambientStrength: Float(ambientStrength),
            saturation: Float(saturation),
            chromaticAberration: Float(chromaticAberration),
            glassColor: SIMD4<Float>(Float(r), Float(g), Float(b), Float(a)),
            thickness: Float(thickness),
            frostIntensity: Float(frostIntensity),
            fresnelIntensity: Float(fresnelIntensity),
            innerShadowIntensity: Float(innerShadowIntensity),
            morphScale: SIMD2<Float>(Float(morphScale.width), Float(morphScale.height))
        )
        renderEncoder.setFragmentBytes(&params, length: MemoryLayout<LiquidGlassV2Params>.stride, index: 0)

        renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        renderEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilScheduled()
        drawable.present()
    }
}

public class LiquidGlassV2View: UIView {
    override public class var layerClass: AnyClass {
        return LiquidGlassV2MetalLayer.self
    }

    var refractionLayer: LiquidGlassV2MetalLayer {
        return layer as! LiquidGlassV2MetalLayer
    }

    // MARK: - Content Views (LiquidLensView compatibility)

    public let contentView: UIView = UIView()
    private let liftedContainerView: UIView = UIView()

    public var selectedContentView: UIView {
        return liftedContainerView
    }

    // Selection lens view
    private var selectionLensView: UIView?
    private var selectionParams: (x: CGFloat, width: CGFloat, isLifted: Bool)?

    public var isDarkThemeOverrided: Bool = false {
        didSet {
            updateForeground()
        }
    }

    public var blurRadius: CGFloat = 0 {
        didSet {
            refractionLayer.blurRadius = blurRadius
        }
    }

    public var curvature: CGFloat = 0.3 {
        didSet {
            refractionLayer.curvature = curvature
        }
    }

    public var domeHeight: CGFloat = 0.02 {
        didSet {
            refractionLayer.domeHeight = domeHeight
        }
    }

    public var maxEdgeBlur: CGFloat = 1.5 {
        didSet {
            refractionLayer.maxEdgeBlur = maxEdgeBlur
        }
    }

    public var lightAngle: CGFloat = .pi / 2 {
        didSet {
            refractionLayer.lightAngle = lightAngle
        }
    }

    public var lightIntensity: CGFloat = 0.5 {
        didSet {
            refractionLayer.lightIntensity = lightIntensity
        }
    }

    public var ambientStrength: CGFloat = 0.3 {
        didSet {
            refractionLayer.ambientStrength = ambientStrength
        }
    }

    public var overlayTint: UIColor = UIColor(white: 1.0, alpha: 0.65) {
        didSet {
            refractionLayer.overlayTint = overlayTint
        }
    }

    public var saturation: CGFloat = 1.6 {
        didSet {
            refractionLayer.saturation = saturation
        }
    }

    public var chromaticAberration: CGFloat = 0.3 {
        didSet {
            refractionLayer.chromaticAberration = chromaticAberration
        }
    }

    public var thickness: CGFloat = 0.03 {
        didSet {
            refractionLayer.thickness = thickness
        }
    }

    public var frostIntensity: CGFloat = 0.8 {
        didSet {
            refractionLayer.frostIntensity = frostIntensity
        }
    }

    public var fresnelIntensity: CGFloat = 0.3 {
        didSet {
            refractionLayer.fresnelIntensity = fresnelIntensity
        }
    }

    public var innerShadowIntensity: CGFloat = 0.15 {
        didSet {
            refractionLayer.innerShadowIntensity = innerShadowIntensity
        }
    }

    public var morphScale: CGSize = CGSize(width: 1.0, height: 1.0) {
        didSet {
            refractionLayer.morphScale = morphScale
        }
    }

    public var innerShadowRadius: CGFloat = 10 {
        didSet {
            updateInnerShadow()
        }
    }

    public var innerShadowOpacity: Float = 0.1 {
        didSet {
            updateInnerShadow()
        }
    }

    public var innerShadowColor: UIColor = .black {
        didSet {
            updateInnerShadow()
        }
    }

    public var innerShadowOffset: CGSize = .zero {
        didSet {
            updateInnerShadow()
        }
    }

    public var overlayEnabled: Bool = true {
        didSet {
            updateForeground()
        }
    }

    private var displayLink: CADisplayLink?

    private var overlayColorView: UIView
    private var innerShadowLayer: CAShapeLayer?
    private var strokeLayer: CAShapeLayer?

    var frameProvider: (() -> (CGRect))?

    public override init(frame: CGRect) {
        overlayColorView = UIView()
        super.init(frame: frame)
        commonSetup()
    }

    public init(
        backgroundContentProvider: @escaping () -> (UIImage?, CGRect)?,
        frameProvider: (() -> (CGRect))?
    ) {
        overlayColorView = UIView()
        self.frameProvider = frameProvider

        super.init(frame: .zero)

        refractionLayer.backgroundContentProvider = backgroundContentProvider
        commonSetup()
    }

    private func commonSetup() {
        backgroundColor = .clear
        isOpaque = false
        clipsToBounds = true

        layer.cornerCurve = .continuous

        addSubview(overlayColorView)
        overlayColorView.backgroundColor = .clear
        overlayColorView.translatesAutoresizingMaskIntoConstraints = false
        overlayColorView.leadingAnchor.constraint(equalTo: leadingAnchor).isActive = true
        overlayColorView.trailingAnchor.constraint(equalTo: trailingAnchor).isActive = true
        overlayColorView.topAnchor.constraint(equalTo: topAnchor).isActive = true
        overlayColorView.bottomAnchor.constraint(equalTo: bottomAnchor).isActive = true

        // Setup content views for LiquidLensView compatibility
        contentView.isUserInteractionEnabled = false
        liftedContainerView.isUserInteractionEnabled = false
        addSubview(contentView)
        addSubview(liftedContainerView)

        setupDisplayLink()
        setupInnerShadow()
        setupStrokeLayer()
        setupSelectionLens()
    }

    private func setupSelectionLens() {
        let lens = UIView()
        lens.backgroundColor = UIColor.white.withAlphaComponent(0.1)
        lens.layer.cornerCurve = .continuous
        insertSubview(lens, at: 0)
        selectionLensView = lens
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private func setupInnerShadow() {
        let shadowLayer = CAShapeLayer()
        shadowLayer.masksToBounds = true
        layer.addSublayer(shadowLayer)
        innerShadowLayer = shadowLayer
    }

    private func setupStrokeLayer() {
        let strokeLayer = CAShapeLayer()
        strokeLayer.masksToBounds = true
        layer.addSublayer(strokeLayer)
        strokeLayer.opacity = 0.5

        self.strokeLayer = strokeLayer
    }

    deinit {
        displayLink?.invalidate()
    }

    private func setupDisplayLink() {
        let displayLink = CADisplayLink(target: self, selector: #selector(self.displayLinkFired))

        if #available(iOS 15.0, *) {
            displayLink.preferredFrameRateRange = CAFrameRateRange(minimum: 60, maximum: 120, preferred: 120)
        }

        displayLink.add(to: .current, forMode: .common)
        self.displayLink = displayLink
    }

    @objc private func displayLinkFired() {
        if let frameProvider {
            CATransaction.begin()
            CATransaction.setDisableActions(true)

            self.frame = frameProvider()

            CATransaction.commit()
        }

        updateInnerShadow()
        updateStrokeLayer()
        refractionLayer.setNeedsDisplay()
    }

    private func updateInnerShadow() {
        guard let shadowLayer = innerShadowLayer else { return }

        let currentBounds = bounds
        let currentCornerRadius = bounds.height / 2.0

        CATransaction.begin()
        CATransaction.setDisableActions(true)

        shadowLayer.frame = currentBounds

        let inset: CGFloat = -innerShadowRadius * 2
        let shadowPath = UIBezierPath(
            roundedRect: currentBounds.insetBy(dx: inset, dy: inset),
            cornerRadius: currentCornerRadius
        )

        let cutout = UIBezierPath(roundedRect: currentBounds.insetBy(dx: -4, dy: -4),
                                  cornerRadius: currentCornerRadius).reversing()

        shadowPath.append(cutout)

        shadowLayer.path = shadowPath.cgPath
        shadowLayer.fillColor = innerShadowColor.cgColor
        shadowLayer.shadowColor = innerShadowColor.cgColor
        shadowLayer.shadowOffset = innerShadowOffset
        shadowLayer.shadowOpacity = innerShadowOpacity
        shadowLayer.shadowRadius = innerShadowRadius

        CATransaction.commit()
    }

    private func updateStrokeLayer() {
        guard let strokeLayer else { return }

        let currentBounds = bounds
        let currentCornerRadius = bounds.height / 2.0

        CATransaction.begin()
        CATransaction.setDisableActions(true)

        strokeLayer.frame = currentBounds

        let path = UIBezierPath(
            roundedRect: currentBounds,
            cornerRadius: currentCornerRadius
        )

        strokeLayer.path = path.cgPath
        strokeLayer.strokeColor = UIColor.white.cgColor
        strokeLayer.lineWidth = 2
        strokeLayer.fillColor = UIColor.clear.cgColor

        let mask = CAGradientLayer()
        mask.frame = currentBounds
        mask.type = .axial
        mask.startPoint = CGPoint(x: 0, y: 0)
        mask.endPoint = CGPoint(x: 1, y: 1)
        mask.colors = [
            UIColor.white.cgColor,
            UIColor.white.withAlphaComponent(0).cgColor,
            UIColor.white.cgColor
        ]

        strokeLayer.mask = mask

        strokeLayer.opacity = isDarkThemeOverrided || traitCollection.userInterfaceStyle == .dark ? 0.5 : 1.0

        CATransaction.commit()
    }

    public override func traitCollectionDidChange(_ previousTraitCollection: UITraitCollection?) {
        updateForeground()
    }

    private func updateForeground() {
        updateStrokeLayer()

        if overlayEnabled {
            overlayColorView.backgroundColor = traitCollection.userInterfaceStyle == .dark || isDarkThemeOverrided ? UIColor.white.withAlphaComponent(0.1) : .clear
        } else {
            overlayColorView.backgroundColor = .clear
        }
    }

    override public func layoutSubviews() {
        super.layoutSubviews()

        updateForeground()

        refractionLayer.frame = bounds

        if bounds.size != .zero {
            refractionLayer.drawableSize = CGSize(
                width: bounds.width * UIScreen.main.scale,
                height: bounds.height * UIScreen.main.scale
            )
        }
        refractionLayer.shapeCornerRadius = layer.cornerRadius

        // Update content views
        contentView.frame = bounds
        liftedContainerView.frame = bounds
    }

    // MARK: - LiquidLensView Compatibility

    public func update(size: CGSize, selectionX: CGFloat, selectionWidth: CGFloat, isDark: Bool, isLifted: Bool, transition: ComponentTransition) {
        self.isDarkThemeOverrided = isDark
        self.selectionParams = (selectionX, selectionWidth, isLifted)

        let cornerRadius = size.height * 0.5
        layer.cornerRadius = cornerRadius

        // Update selection lens
        if let lens = selectionLensView {
            let lensFrame = CGRect(
                x: max(0, min(selectionX, size.width - selectionWidth)),
                y: 0,
                width: selectionWidth,
                height: size.height
            )

            let liftedInset: CGFloat = isLifted ? -4.0 : 0.0
            let effectiveLensFrame = lensFrame.insetBy(dx: liftedInset, dy: liftedInset)

            lens.layer.cornerRadius = effectiveLensFrame.height * 0.5
            lens.backgroundColor = isDark ? UIColor.white.withAlphaComponent(0.15) : UIColor.black.withAlphaComponent(0.05)

            transition.setFrame(view: lens, frame: effectiveLensFrame)
            transition.setScale(view: liftedContainerView, scale: isLifted ? 1.15 : 1.0)
        }

        // Update content views
        transition.setFrame(view: contentView, frame: CGRect(origin: .zero, size: size))
        transition.setFrame(view: liftedContainerView, frame: CGRect(origin: .zero, size: size))
    }

    public var selectionX: CGFloat? {
        return selectionParams?.x
    }

    public var selectionWidth: CGFloat? {
        return selectionParams?.width
    }
}

struct LiquidGlassV2Params {
    var radius: Float
    var aspectRatio: Float
    var uvOffset: SIMD2<Float>
    var uvScale: SIMD2<Float>
    var p: Float
    var height: Float
    var maxEdgeBlur: Float
    var lightAngle: Float
    var lightIntensity: Float
    var ambientStrength: Float
    var saturation: Float
    var chromaticAberration: Float
    var glassColor: SIMD4<Float>
    var thickness: Float
    var frostIntensity: Float
    var fresnelIntensity: Float
    var innerShadowIntensity: Float
    var morphScale: SIMD2<Float>
}
