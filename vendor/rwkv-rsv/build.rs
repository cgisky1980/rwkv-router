use std::path::Path;
use std::process::Command;

// ============================================================
// Types
// ============================================================

/// A single variant of a shader preprocessor define.
struct DefineVariant {
    /// How to emit the -D flag:
    ///   None       → no flag
    ///   Some("")   → flag only (-DNAME)
    ///   Some("v")  → value define (-DNAME=v)
    shader_value: Option<String>,
    /// Component to include in the output filename. Empty = omitted.
    filename_value: &'static str,
}

/// A dimension of shader preprocessor defines (e.g., A_TYPE, ACTIVATION).
struct DefineDimension {
    name: &'static str,
    variants: Vec<DefineVariant>,
}

/// A shader specification with its source file and define dimensions.
struct ShaderSpec {
    name: &'static str,
    source: &'static str,
    defines: Vec<DefineDimension>,
}

// ============================================================
// Variant constructors
// ============================================================

/// Value define: `-DNAME=value`, with a filename component.
///
/// Use an empty `filename` to omit this dimension from the output filename.
fn def<T: std::fmt::Display>(value: T, filename: &'static str) -> DefineVariant {
    DefineVariant {
        shader_value: Some(value.to_string()),
        filename_value: filename,
    }
}

/// Flag define: `-DNAME` (no value), with a filename component.
fn flag(filename: &'static str) -> DefineVariant {
    DefineVariant {
        shader_value: Some(String::new()),
        filename_value: filename,
    }
}

/// No define: no `-D` flag emitted, with a filename component.
fn none(filename: &'static str) -> DefineVariant {
    DefineVariant {
        shader_value: None,
        filename_value: filename,
    }
}

// ============================================================
// Macro for declaring shader compilation specs
// ============================================================

/// Declare shader variants for compilation.
///
/// Usage:
/// ```ignore
/// compile_shaders! {
///     "name" => "src/shader.comp" => {
///         DEFINE_A => [def("value1", "fn1"), def("value2", "fn2")],
///         DEFINE_B => [none("default"), flag("enabled")],
///         DEFINE_C => [none("off"), def(1, "on")],
///     },
/// }
/// ```
///
/// Variant constructors:
/// - `def(value, filename)` — emits `-DDEFINE=value`; `filename` appears in output name
/// - `flag(filename)`       — emits `-DDEFINE` (no value); `filename` appears in output name
/// - `none(filename)`       — emits no `-D` flag; `filename` appears in output name
///
/// Pass an empty string as `filename` to omit that dimension from the output filename.
macro_rules! compile_shaders {
    (
        $(
            $name:literal => $source:literal => {
                $( $define:ident => [ $($variant:expr),+ $(,)? ] ),+ $(,)?
            }
        ),+ $(,)?
    ) => {
        vec![
            $(
                ShaderSpec {
                    name: $name,
                    source: $source,
                    defines: vec![
                        $(
                            DefineDimension {
                                name: stringify!($define),
                                variants: vec![$($variant),+],
                            },
                        )+
                    ],
                },
            )+
        ]
    };
}

// ============================================================
// Compilation logic
// ============================================================

fn cartesian_product(defines: &[DefineDimension]) -> Vec<Vec<(&str, &DefineVariant)>> {
    let mut result: Vec<Vec<(&str, &DefineVariant)>> = vec![vec![]];
    for dim in defines {
        let mut next = Vec::new();
        for existing in &result {
            for variant in &dim.variants {
                let mut combo = existing.clone();
                combo.push((dim.name, variant));
                next.push(combo);
            }
        }
        result = next;
    }
    result
}

fn compile_shader(shaders_dir: &Path, spv_dir: &Path, spec: &ShaderSpec) {
    let source_path = shaders_dir.join(spec.source);
    println!("cargo:rerun-if-changed={}", source_path.display());
    let source_mtime = std::fs::metadata(&source_path)
        .and_then(|m| m.modified())
        .ok();

    for combo in cartesian_product(&spec.defines) {
        let filename_parts: Vec<&str> = combo
            .iter()
            .filter_map(|(_, v)| {
                if v.filename_value.is_empty() {
                    None
                } else {
                    Some(v.filename_value)
                }
            })
            .collect();

        let filename = if filename_parts.is_empty() {
            format!("{}.spv", spec.name)
        } else {
            format!("{}_{}.spv", spec.name, filename_parts.join("_"))
        };

        let output_path = spv_dir.join(&filename);
        let committed = output_path.exists();

        // Skip recompilation when the committed spv is up-to-date.
        // This lets CI (and users without the Vulkan SDK) build from the
        // checked-in binaries; only shader source edits trigger a rebuild.
        if let (Some(src), Ok(out_meta)) = (source_mtime, std::fs::metadata(&output_path))
            && let Ok(out_mtime) = out_meta.modified()
            && out_mtime >= src
        {
            continue;
        }

        let mut cmd = Command::new("glslangValidator");
        cmd.arg("--target-env").arg("spirv1.3");

        for (define_name, variant) in &combo {
            match &variant.shader_value {
                None => {}
                Some(v) if v.is_empty() => {
                    cmd.arg(format!("-D{}", define_name));
                }
                Some(v) => {
                    cmd.arg(format!("-D{}={}", define_name, v));
                }
            }
        }

        cmd.arg("-V").arg(&source_path).arg("-o").arg(&output_path);

        // mtime is unreliable right after `git clone` (every file gets the
        // same checkout timestamp; nanosecond ordering may make the source
        // look newer than the committed spv). When glslangValidator is not
        // installed but a committed spv exists, fall back to it instead of
        // failing the build. Real compilation errors still panic.
        match cmd.status() {
            Ok(status) if status.success() => {}
            Ok(_) => panic!("glslangValidator failed for {}", filename),
            Err(_) if committed => {
                println!(
                    "cargo:warning=glslangValidator not found; using committed {}",
                    filename
                );
            }
            Err(e) => panic!(
                "failed to run glslangValidator for {} — install it (Vulkan SDK / \
                 glslang-tools) or ensure the spv is committed: {}",
                filename, e
            ),
        }
    }
}

// ============================================================
// Main
// ============================================================

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let shaders_dir = Path::new("assets/shaders");
    let spv_dir = shaders_dir.join("spv");
    std::fs::create_dir_all(&spv_dir).unwrap();

    let specs = compile_shaders! {
        "gemm" => "src/gemm.comp" => {
            A_BITS => [def(16, "")],
            A_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            C_BITS => [def(32, ""), def(16, "")],
            C_TYPE => [def("float32_t", "f32"), def("float16_t", "f16")],
            AFFINE => [none(""), flag("affine")],
            ACTIVATION => [none(""), def(1, "relu2"), def(2, "tanh")],
        },
        "gemm_bias" => "src/gemm_bias.comp" => {
            TYPE => [none("")],
        },
        "norm" => "src/norm.comp" => {
            I_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            O_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            AFFINE => [none(""), flag("affine")],
        },
        "gemv" => "src/gemv.comp" => {
            I_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            O_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            AFFINE => [none(""), flag("affine")],
            ACTIVATION => [none(""), def(1, "relu2"), def(2, "tanh")],
        },
        "gemv_f32io" => "src/gemv_f32io.comp" => {
            AFFINE => [none(""), flag("affine")],
            ACTIVATION => [none(""), def(1, "relu2"), def(2, "tanh")],
        },
        "gemv_f32io_mul" => "src/gemv_f32io_mul.comp" => {
            TYPE => [none("")],
        },
        "gemv_f32io_add" => "src/gemv_f32io_add.comp" => {
            MUL => [def(0, ""), def(1, "mul")],
        },
        "gemv_rkv_f32io" => "src/gemv_rkv_f32io.comp" => {
            TYPE => [none("")],
        },
        "gemv_rkv_stage1" => "src/gemv_rkv_stage1.comp" => {
            TYPE => [none("")],
        },
        "gemv_int8" => "src/gemv_int8.comp" => {
            ACTIVATION => [none(""), def(1, "relu2")],
        },
        "gemv_int8_add" => "src/gemv_int8_add.comp" => {
            MUL => [def(0, ""), def(1, "mul")],
        },
        "ffn_value_sparse_add" => "src/ffn_value_sparse_add.comp" => {
            TYPE => [none("")],
        },
        "gemv_int8_rkv_stage1" => "src/gemv_int8_rkv_stage1.comp" => {
            TYPE => [none("")],
        },
        "dequant_int8_f16" => "src/dequant_int8_f16.comp" => {
            TYPE => [none("")],
        },
        "norm_lerp6" => "src/norm_lerp6.comp" => {
            TYPE => [none("")],
        },
        "cmix_norm_lerp" => "src/cmix_norm_lerp.comp" => {
            TYPE => [none("")],
        },
        "gemv_lowrank_chain" => "src/gemv_lowrank_chain.comp" => {
            OP => [def(0, "w"), def(1, "a"), def(2, "v"), def(3, "g")],
        },
        "gemv_lowrank_chain4" => "src/gemv_lowrank_chain4.comp" => {
            TYPE => [none("")],
        },
        "gemv_lowrank_stage1" => "src/gemv_lowrank_stage1.comp" => {
            TYPE => [none("")],
        },
        "token_shift" => "src/token_shift.comp" => {
            I_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            O_TYPE => [def("float16_t", "f16"), def("float32_t", "f32")],
            REVERSED => [none(""), flag("rev")],
        },
        "elementwise" => "src/elementwise.comp" => {
            I_TYPE => [def("float32_t", "f32")],
            O_TYPE => [def("float32_t", "f32")],
            OP => [
                def(1, "sigmoid"),
                def(2, "exp"),
                def(3, "tanh"),
                def(4, "neg"),
                def(5, "mul"),
                def(6, "add"),
                def(7, "lerp"),
                def(8, "scale"),
                def(9, "scale_exp"),
                def(10, "mul_neg"),
            ],
        },
        "dplr" => "src/dplr.comp" => {
            I_TYPE => [def("float32_t", "f32")],
            O_TYPE => [def("float32_t", "f32")],
        },
        "dplr_seq" => "src/dplr_seq.comp" => {
            I_TYPE => [def("float32_t", "f32")],
            O_TYPE => [def("float32_t", "f32")],
        },
        "seq_shift" => "src/seq_shift.comp" => {
            I_TYPE => [def("float32_t", "f32")],
            O_TYPE => [def("float32_t", "f32")],
        },
        "v_first_lerp" => "src/v_first_lerp.comp" => {
            I_TYPE => [def("float32_t", "f32")],
            O_TYPE => [def("float32_t", "f32")],
        },
        "copy_token" => "src/copy_token.comp" => {
            I_TYPE => [def("float32_t", "f32")],
            O_TYPE => [def("float32_t", "f32")],
        },
        "to_f16" => "src/to_f16.comp" => {
            TYPE => [none("")],
        },
        "to_f16_triple" => "src/to_f16_triple.comp" => {
            TYPE => [none("")],
        },
        "l2_norm" => "src/l2_norm.comp" => {
            I_TYPE => [def("float32_t", "f32")],
            O_TYPE => [def("float32_t", "f32")],
        },
        "fuse_ka" => "src/fuse_ka.comp" => {
            TYPE => [none("")],
        },
        "fuse_ka_dplr" => "src/fuse_ka_dplr.comp" => {
            TYPE => [none("")],
        },
        "fuse_ka_dplr_norm" => "src/fuse_ka_dplr_norm.comp" => {
            TYPE => [none("")],
        },
        "sum_rk_rk" => "src/sum_rk_rk.comp" => {
            TYPE => [none("")],
        },
        "norm_sum_rk_rk" => "src/norm_sum_rk_rk.comp" => {
            TYPE => [none("")],
        },
        "argmax" => "src/argmax.comp" => {
            TYPE => [none("")],
        },
        "sample" => "src/sample.comp" => {
            TYPE => [none("")],
        },
        "gather_row" => "src/gather_row.comp" => {
            TYPE => [none("")],
        },
        "gather_row_f16" => "src/gather_row_f16.comp" => {
            TYPE => [none("")],
        },
        "record_token" => "src/record_token.comp" => {
            TYPE => [none("")],
        },
    };

    for spec in &specs {
        compile_shader(shaders_dir, &spv_dir, spec);
    }
}
