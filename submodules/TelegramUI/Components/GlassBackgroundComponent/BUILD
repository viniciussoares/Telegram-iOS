load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")
load(
    "@build_bazel_rules_apple//apple:resources.bzl",
    "apple_resource_bundle",
)
load("//build-system/bazel-utils:plist_fragment.bzl",
    "plist_fragment",
)

filegroup(
    name = "GlassBackgroundComponentMetalResources",
    srcs = glob([
        "MetalResources/**/*.*",
    ]),
    visibility = ["//visibility:public"],
)

plist_fragment(
    name = "GlassBackgroundComponentBundleInfoPlist",
    extension = "plist",
    template =
    """
    <key>CFBundleIdentifier</key>
    <string>org.telegram.GlassBackgroundComponent</string>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleName</key>
    <string>GlassBackgroundComponent</string>
    """
)

apple_resource_bundle(
    name = "GlassBackgroundComponentBundle",
    infoplists = [
        ":GlassBackgroundComponentBundleInfoPlist",
    ],
    resources = [
        ":GlassBackgroundComponentMetalResources",
    ],
)

swift_library(
    name = "GlassBackgroundComponent",
    module_name = "GlassBackgroundComponent",
    srcs = glob([
        "Sources/**/*.swift",
    ]),
    copts = [
        "-warnings-as-errors",
    ],
    data = [
        ":GlassBackgroundComponentBundle",
    ],
    deps = [
        "//submodules/Display",
        "//submodules/ComponentFlow",
        "//submodules/Components/ComponentDisplayAdapters",
        "//submodules/UIKitRuntimeUtils",
        "//submodules/AppBundle",
    ],
    visibility = [
        "//visibility:public",
    ],
)
