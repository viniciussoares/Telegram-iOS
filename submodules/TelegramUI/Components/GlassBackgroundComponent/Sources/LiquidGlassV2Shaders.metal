//
//  LiquidGlassShaders.metal
//  LiquidGlassDemo
//
//  Refraction-based glass effect shader with SDF border distortion
//

#include <metal_stdlib>
using namespace metal;

struct ShaderInput {
    float2 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct FragmentData {
    float4 position [[position]];
    float2 texCoord;
};

struct LiquidGlassV2Params {
    float radius;
    float aspectRatio;
    float2 uvOffset;
    float2 uvScale;
    float p;
    float height;
    float max_edge_blur;
    float lightAngle;
    float lightIntensity;
    float ambientStrength;
    float saturation;           // 1.0 = normal, >1 = more vibrant, <1 = desaturated
    float chromaticAberration;  // 0 = none, higher = more color separation
    float4 glassColor;          // RGBA tint color
    float thickness;            // Glass thickness for dome height profile
    float frostIntensity;       // Uniform background blur (0 = clear, 1 = fully frosted)
    float fresnelIntensity;     // Edge highlight intensity (0 = off, 0.3 = subtle, 1.0 = strong)
    float innerShadowIntensity; // Inner shadow darkness (0 = off, 0.15 = subtle, 0.5 = strong)
    float2 morphScale;          // Shape stretch (1,1 = normal, >1 = larger, <1 = smaller)
};

struct SurfacePoint {
    float2 displacement;        // UV displacement from refraction
    float domeHeight;           // Height at this point on the glass dome
    float borderProximity;      // 0 at center, 1 at edges
};

vertex FragmentData liquidGlassV2Vertex(ShaderInput in [[stage_in]]) {
    FragmentData out;
    out.position = float4(in.position, 0.0, 1.0);
    out.texCoord = in.texCoord;
    return out;
}

// MARK: - SDF Functions for Border Distortion

// Signed Distance Function for a rounded rectangle
// Returns negative inside, 0 at boundary, positive outside
// p: point to evaluate, b: half-extents (width/2, height/2), r: corner radius
float sdRoundedBox(float2 p, float2 b, float r) {
    float2 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

// Height function - creates dome shape from signed distance
// At edges (sd=0): height = 0
// At center (sd=-thickness): height = thickness
// Creates a smooth hemispherical profile
float getDomeHeight(float sd, float thickness) {
    if (sd >= 0.0 || thickness <= 0.0) {
        return 0.0;
    }
    if (sd < -thickness) {
        return thickness;
    }

    // Hemispherical dome: h = sqrt(r² - x²) where x = thickness + sd
    float x = thickness + sd;
    return sqrt(max(0.0, thickness * thickness - x * x));
}

// Calculate surface normal from SDF
// Uses the gradient of the SDF and height to create a 3D normal
// At edges: normal points outward (XY plane)
// At center: normal points up (Z axis)
float3 getSurfaceNormal(float sd, float2 gradient, float thickness) {
    if (thickness <= 0.0 || sd >= 0.0) {
        return float3(0.0, 0.0, 1.0);
    }

    // Clamp sd to -thickness
    float clampedSd = max(sd, -thickness);

    // n_cos: how much the normal points in XY plane (edge factor)
    // n_sin: how much the normal points in Z (upward)
    float n_cos = (thickness + clampedSd) / thickness;
    float n_sin = sqrt(max(0.0, 1.0 - n_cos * n_cos));

    // Scale XY components by n_cos, Z by n_sin
    return normalize(float3(gradient * n_cos, n_sin));
}

// Calculate refraction displacement using Snell's law
float2 calculateRefraction(float3 normal, float height, float thickness, float refractiveIndex) {
    if (height <= 0.0 && normal.z >= 0.999) {
        return float2(0.0);
    }

    // Incident ray coming from camera (looking down Z axis)
    float3 incident = float3(0.0, 0.0, -1.0);

    // Calculate refracted ray using Snell's law
    float eta = 1.0 / refractiveIndex;
    float3 refracted = refract(incident, normal, eta);

    // If total internal reflection, use reflected instead
    if (length(refracted) < 0.001) {
        refracted = reflect(incident, normal);
    }

    // Calculate displacement based on glass thickness
    // Ray travels through glass and exits at bottom
    float baseHeight = thickness * 8.0;
    float rayLength = (height + baseHeight) / max(0.001, abs(refracted.z));

    return refracted.xy * rayLength;
}

// Analytical gradient of rounded rectangle SDF
// Returns normalized direction pointing away from shape (toward nearest edge)
// Much faster than finite differences - no extra SDF evaluations needed
float2 sdfGradientAnalytical(float2 position, float2 halfExtents, float cornerRadius) {
    float2 q = abs(position) - halfExtents + cornerRadius;
    float2 signP = sign(position);

    float2 grad;
    if (q.x > 0.0 && q.y > 0.0) {
        // Corner region: gradient points away from corner arc center
        float len = length(q);
        grad = (len > 0.0001) ? (q / len) : float2(0.707, 0.707);
        grad *= signP;
    } else if (q.x > q.y) {
        // Vertical edge region: gradient points left/right
        grad = float2(signP.x, 0.0);
    } else {
        // Horizontal edge region: gradient points up/down
        grad = float2(0.0, signP.y);
    }

    return grad;
}

// Compute surface refraction data using SDF-based dome geometry
// Full border distortion around the entire rounded rectangle perimeter
SurfacePoint computeSurfaceRefraction(float2 uv, float aspectRatio, float cornerRadius,
                                       float thickness, float wavelength, float dispersion,
                                       float2 morphScale) {
    SurfacePoint point;

    // Convert UV (0-1) to centered coordinates (-0.5 to 0.5)
    float2 centeredUV = uv - 0.5;
    centeredUV.y *= aspectRatio;

    // Apply morph scale - dividing stretches the shape visually
    // morphScale > 1 makes shape appear larger, < 1 makes it smaller
    float2 scaledUV = centeredUV / morphScale;

    // Half-extents of the rounded rectangle
    float2 halfExtents = float2(0.5, aspectRatio * 0.5);

    // Calculate signed distance using scaled coordinates
    float sd = sdRoundedBox(scaledUV, halfExtents, cornerRadius);

    // Get dome height at this point
    float height = getDomeHeight(sd, thickness);
    point.domeHeight = height;

    // Calculate border proximity (1 at edge, 0 at center)
    float normalizedHeight = thickness > 0.0 ? height / thickness : 0.0;
    float edgeThreshold = 0.7; // How far from edge the effect extends
    point.borderProximity = 1.0 - smoothstep(0.0, edgeThreshold, normalizedHeight);

    // Outside the shape - no refraction
    if (sd >= 0.0) {
        point.displacement = float2(0.0);
        point.borderProximity = 0.0;
        return point;
    }

    // Calculate SDF gradient analytically (direction to nearest edge)
    float2 gradient = sdfGradientAnalytical(scaledUV, halfExtents, cornerRadius);

    // Get surface normal from SDF
    float3 normal = getSurfaceNormal(sd, gradient, thickness);

    // Calculate refractive index with chromatic dispersion
    // Reduce dispersion at edges to prevent color shifts
    float baseIndex = 1.8;
    float refractiveIndex = baseIndex;
    float edgeDispersionScale = 1.0 - point.borderProximity * 0.9;
    float effectiveDispersion = dispersion * edgeDispersionScale;
    if (effectiveDispersion > 0.001) {
        float wavelengthSq = wavelength * wavelength;
        float wavelengthQuad = wavelengthSq * wavelengthSq;
        float B = effectiveDispersion * 0.08 * (baseIndex - 1.0);
        float C = effectiveDispersion * 0.003 * (baseIndex - 1.0);
        refractiveIndex = baseIndex - B / wavelengthSq - C / wavelengthQuad;
    }

    // Calculate refraction offset
    float2 offset = calculateRefraction(normal, height, thickness, refractiveIndex);

    // Scale offset back to UV space
    point.displacement = offset / float2(1.0, aspectRatio);

    return point;
}

float normalDistWeight(float dist, float spread) {
    return exp(-(dist * dist) / (2.0 * spread * spread));
}

// MARK: - Rim Lighting Functions

constant float3 LUMA_WEIGHTS = float3(0.299, 0.587, 0.114);

float3 getHighlightColor(float3 backgroundColor, float targetBrightness) {
    float luminance = dot(backgroundColor, LUMA_WEIGHTS);

    // Calculate saturation
    float maxComponent = max(max(backgroundColor.r, backgroundColor.g), backgroundColor.b);
    float minComponent = min(min(backgroundColor.r, backgroundColor.g), backgroundColor.b);
    float saturation = maxComponent > 0.0 ? (maxComponent - minComponent) / maxComponent : 0.0;

    // Create a colored highlight
    float3 coloredHighlight = float3(targetBrightness);

    if (luminance > 0.001) {
        float3 normalizedBackground = backgroundColor / luminance;
        coloredHighlight = normalizedBackground * targetBrightness;

        // Boost saturation for more vivid highlights
        float saturationBoost = 1.3;
        float3 gray = float3(dot(coloredHighlight, LUMA_WEIGHTS));
        coloredHighlight = mix(gray, coloredHighlight, saturationBoost);
        coloredHighlight = min(coloredHighlight, float3(1.0));
    }

    // Blend towards white based on darkness and saturation
    float luminanceFactor = smoothstep(0.0, 0.6, luminance);
    float saturationFactor = smoothstep(0.0, 0.4, saturation);
    float colorInfluence = luminanceFactor * saturationFactor;

    float3 whiteHighlight = float3(targetBrightness);
    return mix(whiteHighlight, coloredHighlight, colorInfluence);
}

half4 applyGlassColor(half4 liquidColor, float4 glassColor) {
    if (glassColor.a < 0.001) {
        return liquidColor;
    }

    half4 finalColor = liquidColor;
    half3 glassRGB = half3(glassColor.rgb);
    half glassAlpha = half(glassColor.a);
    half glassLuminance = dot(glassRGB, half3(LUMA_WEIGHTS));

    if (glassLuminance < 0.5h) {
        // Dark tint: multiply blend
        half3 darkened = liquidColor.rgb * (glassRGB * 2.0h);
        finalColor.rgb = mix(liquidColor.rgb, darkened, glassAlpha);
    } else {
        // Light tint: screen blend
        half3 invLiquid = half3(1.0h) - liquidColor.rgb;
        half3 invGlass = half3(1.0h) - glassRGB;
        half3 screened = half3(1.0h) - (invLiquid * invGlass);
        finalColor.rgb = mix(liquidColor.rgb, screened, glassAlpha);
    }

    return finalColor;
}

half3 applySaturation(half3 color, float saturation) {
    half lum = dot(color, half3(LUMA_WEIGHTS));
    half3 saturatedColor = mix(half3(lum), color, half(saturation));
    return clamp(saturatedColor, 0.0h, 1.0h);
}

// Fresnel effect - bright highlight at glass edges
// Based on edge factor with power falloff for sharp edge appearance
half calculateFresnel(float edgeFactor, float intensity) {
    if (intensity < 0.001 || edgeFactor < 0.01) {
        return 0.0h;
    }

    // Sharper falloff at edges using power function
    // edgeFactor is 1 at edge, 0 at center
    float fresnel = pow(edgeFactor, 2.0);  // Square for sharper edge

    // Scale by intensity
    return half(fresnel * intensity * 0.7);
}

// Inner shadow - darkening near edges on the opposite side of light
// Creates depth by simulating light being blocked at glass edges
half calculateInnerShadow(float edgeFactor, float2 gradientDir, float2 lightDirection, float intensity) {
    if (intensity < 0.001 || edgeFactor < 0.01) {
        return 0.0h;
    }

    // Shadow appears opposite to light direction
    // gradientDir points toward the edge, lightDirection points toward light
    // Shadow is strongest where gradient points away from light (dot product negative)
    float2 shadowDir = -lightDirection;
    float dirFactor = max(0.0, dot(normalize(gradientDir), shadowDir));

    // Sharpen the directional falloff
    dirFactor = pow(dirFactor, 1.5);

    // Combine edge proximity with directional factor
    float shadow = edgeFactor * dirFactor;

    return half(shadow * intensity);
}

half3 calculateRimLighting(float2 refractionVector, float edgeFactor, float2 lightDirection,
                           float lightIntensity, float ambientStrength, half3 backgroundColor) {
    if (edgeFactor < 0.01 || lightIntensity < 0.01) {
        return half3(0.0h);
    }

    // Calculate normal direction from refraction vector
    float2 normalXY = normalize(refractionVector);

    // Main directional light
    half mainLightInfluence = half(max(0.0, dot(normalXY, lightDirection)));

    // Secondary light from opposite direction
    half oppositeLightInfluence = half(max(0.0, dot(normalXY, -lightDirection)));

    // Combine with opposite light at 80% strength
    half totalInfluence = mainLightInfluence + oppositeLightInfluence * 0.8h;

    // Get highlight color based on background
    half3 highlightColor = half3(getHighlightColor(float3(backgroundColor), 1.0));

    // Directional rim component
    half3 directionalRim = (highlightColor * 0.7h) * (totalInfluence * totalInfluence) * half(lightIntensity) * 2.0h;

    // Ambient rim component
    half3 ambientRim = (highlightColor * 0.4h) * half(ambientStrength);

    // Combine and apply edge factor for rim falloff
    return (directionalRim + ambientRim) * half(edgeFactor);
}

fragment half4 liquidGlassV2Fragment(FragmentData in [[stage_in]],
                                    texture2d<half> backgroundTexture [[texture(0)]],
                                    constant LiquidGlassV2Params &params [[buffer(0)]]) {

    float2 uv = float2(in.texCoord);

    float cornerRadius = params.radius;
    float aspectRatio = params.aspectRatio;
    float chromaticAberration = params.chromaticAberration;
    bool hasChromatic = chromaticAberration > 0.001;

    // Wavelengths in micrometers for Cauchy dispersion
    const float red_wavelength = 0.65;
    const float green_wavelength = 0.55;
    const float blue_wavelength = 0.45;

    // Glass thickness controls the dome profile
    // Use the new thickness param, falling back to height * p for compatibility
    float thickness = params.thickness > 0.0 ? params.thickness : params.height * params.p * 0.1;

    // Calculate green (center wavelength) first using SDF-based approach
    SurfacePoint greenSurface = computeSurfaceRefraction(uv, aspectRatio, cornerRadius, thickness, green_wavelength, chromaticAberration, params.morphScale);
    float2 greenDisp = greenSurface.displacement;

    // Early exit: only if no frost AND no refraction AND no chromatic
    bool hasFrost = params.frostIntensity > 0.01;
    if (length(greenDisp) < 0.0001 && !hasChromatic && greenSurface.borderProximity < 0.01 && !hasFrost) {
        float2 fixed_uv = params.uvOffset + uv * params.uvScale;
        constexpr sampler textureSampler(mag_filter::bicubic, min_filter::bicubic);
        return backgroundTexture.sample(textureSampler, fixed_uv);
    }

    // Only calculate red/blue channels if chromatic aberration is enabled
    float2 redDisp, blueDisp;
    if (hasChromatic) {
        SurfacePoint redSurface = computeSurfaceRefraction(uv, aspectRatio, cornerRadius, thickness, red_wavelength, chromaticAberration, params.morphScale);
        SurfacePoint blueSurface = computeSurfaceRefraction(uv, aspectRatio, cornerRadius, thickness, blue_wavelength, chromaticAberration, params.morphScale);
        redDisp = redSurface.displacement;
        blueDisp = blueSurface.displacement;
    } else {
        redDisp = greenDisp;
        blueDisp = greenDisp;
    }

    constexpr sampler textureSampler(mag_filter::bicubic, min_filter::bicubic);

    // Calculate light direction from angle
    float2 lightDirection = float2(cos(params.lightAngle), sin(params.lightAngle));

    // Use the SDF-computed border proximity for lighting
    float borderProx = greenSurface.borderProximity;

    // Frost blur: uniform blur across the entire glass surface (0-20 range)
    float frostBlur = params.frostIntensity * 20.0;

    // Edge blur: adaptive based on refraction magnitude
    float dispMag = length(greenDisp);
    float edgeBlur = min(params.max_edge_blur, dispMag * 25.0);

    // Combine: frost provides base blur, edge blur adds to it at edges
    float blurAmount = max(frostBlur, edgeBlur);

    half4 finalColor;
    float2 fixed_uv = params.uvOffset + uv * params.uvScale;

    if (blurAmount < 0.00001) {
        // No blur path - direct sampling with refraction
        float2 redTexUV = fixed_uv + redDisp * params.uvScale;
        float2 greenTexUV = fixed_uv + greenDisp * params.uvScale;
        float2 blueTexUV = fixed_uv + blueDisp * params.uvScale;

        half4 redSample = backgroundTexture.sample(textureSampler, redTexUV);
        half4 greenSample = backgroundTexture.sample(textureSampler, greenTexUV);
        half4 blueSample = backgroundTexture.sample(textureSampler, blueTexUV);

        finalColor = half4(
            redSample.r,
            greenSample.g,
            blueSample.b,
            (redSample.a + greenSample.a + blueSample.a) / 3.0h
        );
    } else {
        // Blur path with SDF-based refraction
        float2 texelSize = 1.0 / float2(backgroundTexture.get_width(), backgroundTexture.get_height());
        // Correct for aspect ratio so blur is circular, not stretched horizontally
        float2 correctedTexel = texelSize * float2(1.0, aspectRatio);
        half4 accumColor = half4(0);
        half weightSum = 0;

        int kernelRadius = int(ceil(blurAmount * 3.0));
        int stepSize = int(ceil(float(kernelRadius) / 1.5));
        int diameter = 2 * kernelRadius + 1;
        int sampleCount = diameter * diameter;
        int centerIdx = (2 * kernelRadius + 2) * kernelRadius;
        int startIdx = centerIdx % stepSize;

        for (int idx = startIdx; idx < sampleCount; idx += stepSize) {
            int x = idx % diameter - kernelRadius;
            int y = idx / diameter - kernelRadius;

            float dist = length(float2(x, y));
            half w = half(normalDistWeight(dist, blurAmount));
            float2 sampleOffset = float2(x, y) * correctedTexel;
            float2 sampleUV = uv + sampleOffset;
            float2 sampleFixedUV = params.uvOffset + sampleUV * params.uvScale;

            if (hasChromatic) {
                // Chromatic dispersion path with SDF
                SurfacePoint rPt = computeSurfaceRefraction(sampleUV, aspectRatio, cornerRadius, thickness, red_wavelength, chromaticAberration, params.morphScale);
                SurfacePoint gPt = computeSurfaceRefraction(sampleUV, aspectRatio, cornerRadius, thickness, green_wavelength, chromaticAberration, params.morphScale);
                SurfacePoint bPt = computeSurfaceRefraction(sampleUV, aspectRatio, cornerRadius, thickness, blue_wavelength, chromaticAberration, params.morphScale);

                float2 redTexUV = sampleFixedUV + rPt.displacement * params.uvScale;
                float2 greenTexUV = sampleFixedUV + gPt.displacement * params.uvScale;
                float2 blueTexUV = sampleFixedUV + bPt.displacement * params.uvScale;

                half4 redSample = backgroundTexture.sample(textureSampler, redTexUV);
                half4 greenSample = backgroundTexture.sample(textureSampler, greenTexUV);
                half4 blueSample = backgroundTexture.sample(textureSampler, blueTexUV);

                accumColor.r += redSample.r * w;
                accumColor.g += greenSample.g * w;
                accumColor.b += blueSample.b * w;
                accumColor.a += (redSample.a + greenSample.a + blueSample.a) / 3.0h * w;
            } else {
                // Fast path - single SDF call when no chromatic dispersion
                SurfacePoint pt = computeSurfaceRefraction(sampleUV, aspectRatio, cornerRadius, thickness, green_wavelength, 0.0, params.morphScale);
                float2 texUV = sampleFixedUV + pt.displacement * params.uvScale;
                half4 texSample = backgroundTexture.sample(textureSampler, texUV);
                accumColor += texSample * w;
            }
            weightSum += w;
        }

        finalColor = accumColor / weightSum;
    }

    // Apply glass color tinting
    finalColor = applyGlassColor(finalColor, params.glassColor);

    // Apply saturation adjustment
    finalColor.rgb = applySaturation(finalColor.rgb, params.saturation);

    // Apply inner shadow (darkening on opposite side of light)
    // Use displacement vector as proxy for gradient direction
    half innerShadow = calculateInnerShadow(borderProx, greenDisp, lightDirection, params.innerShadowIntensity);
    finalColor.rgb = mix(finalColor.rgb, half3(0.0h), innerShadow);

    // Apply enhanced rim lighting using SDF border proximity
    half3 rimLight = calculateRimLighting(
        greenDisp,
        borderProx,
        lightDirection,
        params.lightIntensity,
        params.ambientStrength,
        finalColor.rgb
    );

    finalColor.rgb += rimLight;

    // Apply Fresnel highlight at edges (bright white edge glow)
    half fresnel = calculateFresnel(borderProx, params.fresnelIntensity);
    finalColor.rgb = mix(finalColor.rgb, half3(1.0h), fresnel);

    return finalColor;
}
