use std::env;
use std::path::PathBuf;
use std::process::{Command, Output};

fn main() {
    // Allow custom cfg used in src/main.rs without triggering unexpected_cfgs.
    println!("cargo:rustc-check-cfg=cfg(ptx_only)");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let fatbin_path = out_dir.join("kernel.fatbin");
    let ptx_path = out_dir.join("kernel.ptx");

    println!("cargo:warning=Compiling CUDA kernel...");

    let ptx_only = env::var("CUDA_PTX_ONLY").ok().as_deref() == Some("1");

    let mut base_args: Vec<String> = Vec::new();
    if ptx_only {
        base_args.push("--ptx".to_string());
        base_args.push("--use_fast_math".to_string());
        base_args.push("-arch=compute_120".to_string());
        base_args.push("-code=compute_120".to_string());
        println!("cargo:rustc-cfg=ptx_only");
    } else {
        base_args.push("--fatbin".to_string());
        base_args.push("--use_fast_math".to_string());
        // SASSを含めてPTX JIT失敗を避ける（T4/A100/30xx/40xx/90系）
        base_args.push("-gencode".to_string());
        base_args.push("arch=compute_75,code=sm_75".to_string());
        base_args.push("-gencode".to_string());
        base_args.push("arch=compute_80,code=sm_80".to_string());
        base_args.push("-gencode".to_string());
        base_args.push("arch=compute_86,code=sm_86".to_string());
        base_args.push("-gencode".to_string());
        base_args.push("arch=compute_89,code=sm_89".to_string());
    }

    // CUDA 12+ならH100向けSM90も追加
    if !ptx_only {
        if let Ok(output) = Command::new("nvcc").arg("--version").output() {
            let text = String::from_utf8_lossy(&output.stdout);
            if text.contains("release 12.") || text.contains("release 13.") {
                base_args.push("-gencode".to_string());
                base_args.push("arch=compute_90,code=sm_90".to_string());
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Windows/MSVC環境では、nvccにホストコンパイラのパスを明示的に指定する必要がある場合がある
        let msvc_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64";
        base_args.push("-ccbin".to_string());
        base_args.push(msvc_path.to_string());
    }

    // RTX 5090 (SM120) 対応: nvccが対応していればPTXを含める。
    // CUDA_SM120=1 で強制、CUDA_SM120=0 で無効化、未指定なら自動検出。
    let sm120_mode = env::var("CUDA_SM120").ok();
    let try_sm120 = match sm120_mode.as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => !ptx_only,
    };

    let mut sm120_args: Vec<String> = Vec::new();
    if try_sm120 && !ptx_only {
        sm120_args.push("-gencode".to_string());
        sm120_args.push("arch=compute_120,code=sm_120".to_string());
        sm120_args.push("-gencode".to_string());
        sm120_args.push("arch=compute_120,code=compute_120".to_string());
    }

    // B200 (SM100) 対応: nvccが対応していればPTXを含める。
    // CUDA_SM100=1 で強制、CUDA_SM100=0 で無効化、未指定なら自動検出。
    let sm100_mode = env::var("CUDA_SM100").ok();
    let try_sm100 = match sm100_mode.as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => !ptx_only,
    };

    let mut sm100_args: Vec<String> = Vec::new();
    if try_sm100 && !ptx_only {
        sm100_args.push("-gencode".to_string());
        sm100_args.push("arch=compute_100,code=sm_100".to_string());
        sm100_args.push("-gencode".to_string());
        sm100_args.push("arch=compute_100,code=compute_100".to_string());
    }

    let output_path = if ptx_only { &ptx_path } else { &fatbin_path };
    let mut extra_args = Vec::new();
    extra_args.extend_from_slice(&sm120_args);
    extra_args.extend_from_slice(&sm100_args);
    let output = run_nvcc(&base_args, &extra_args, output_path);
    if !output.status.success() && !ptx_only {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let unsupported = stderr.contains("Unsupported gpu architecture")
            || stderr.contains("invalid value for --gpu-architecture")
            || stderr.contains("invalid value for --gpu-name");
        if unsupported && sm120_mode.as_deref() != Some("1") {
            println!("cargo:warning=SM120 not supported by nvcc; building without RTX 5090 PTX.");
            // Retry without SM120 and SM100 extras.
            let output2 = run_nvcc(&base_args, &[], output_path);
            if !output2.status.success() {
                panic!(
                    "nvcc failed without SM120:\n{}",
                    String::from_utf8_lossy(&output2.stderr)
                );
            }
        } else {
            panic!("nvccがエラーコードで終了しました:\n{}", stderr);
        }
    } else if !output.status.success() {
        panic!(
            "nvccがエラーコードで終了しました:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    println!("cargo:rerun-if-changed=src/kernel.cu");
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(target_os = "windows")]
    {
        if let Ok(cuda_path) = env::var("CUDA_HOME") {
            println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
        } else if let Ok(cuda_path) = env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
        } else {
            // 一般的なパスを探索
            let paths = vec![
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/lib/x64",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/lib/x64",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64",
            ];
            for path in paths {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(cuda_path) = env::var("CUDA_HOME") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        } else if let Ok(cuda_path) = env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        } else {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        }
    }

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
}

fn run_nvcc(base_args: &[String], extra_args: &[String], output_path: &PathBuf) -> Output {
    let mut cmd = Command::new("nvcc");
    for a in base_args {
        cmd.arg(a);
    }
    for a in extra_args {
        cmd.arg(a);
    }
    cmd.arg("src/kernel.cu")
        .arg("-o")
        .arg(output_path)
        .output()
        .expect("nvccの実行に失敗しました。CUDA Toolkitがインストールされ、PATHが通っているか確認してください。")
}
