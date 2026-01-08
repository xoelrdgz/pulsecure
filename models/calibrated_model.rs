//! Calibrated quantized model for FHE inference.
//! Auto-generated - DO NOT EDIT.
//!
//! Medical-grade CVD risk prediction.
//! Includes isotonic calibration for accurate probabilities.
//!
//! Validation metrics:
//!   AUC (calibrated): 0.8476
//!   AUC (quantized):  0.8468
//!   Brier score:      0.0651
//!   Max calib error:  20.26%
//!   Recall @ thresh:  97.56%

/// Scale factor (2^12)
pub const SCALE_FACTOR: i64 = 4096;

/// Precision bits
pub const PRECISION_BITS: u32 = 12;

/// Feature names
pub const FEATURE_NAMES: [&str; 9] = [
    "RIDAGEYR",
    "BPQ020",
    "BPXSY1",
    "SMQ020",
    "LBDHDD",
    "LBXSCR",
    "BMXWAIST",
    "DIQ010",
    "LBXGH",
];

/// Quantized coefficients
pub const COEFFICIENTS_Q: [i64; 9] = [
    5661,  // RIDAGEYR
    -2077,  // BPQ020
    -1246,  // BPXSY1
    -969,  // SMQ020
    -1008,  // LBDHDD
    491,  // LBXSCR
    494,  // BMXWAIST
    -431,  // DIQ010
    234,  // LBXGH
];

/// Quantized intercept
pub const INTERCEPT_Q: i64 = -4191;

/// Scaler mean (quantized)
pub const SCALER_MEAN_Q: [i64; 9] = [
    190570,  // RIDAGEYR
    6820,  // BPQ020
    499472,  // BPXSY1
    6490,  // SMQ020
    216551,  // LBDHDD
    3690,  // LBXSCR
    400891,  // BMXWAIST
    7866,  // DIQ010
    23223,  // LBXGH
];

/// Scaler std inverse (quantized)
pub const SCALER_STD_INV_Q: [i64; 9] = [
    226,  // RIDAGEYR
    7749,  // BPQ020
    234,  // BPXSY1
    8006,  // SMQ020
    259,  // LBDHDD
    8985,  // LBXSCR
    250,  // BMXWAIST
    10235,  // DIQ010
    4076,  // LBXGH
];

// === Sigmoid LUT ===
pub const SIGMOID_LUT_INPUT_BITS: u32 = 8;
pub const SIGMOID_LUT_OUTPUT_BITS: u32 = 12;
pub const SIGMOID_INPUT_RANGE: f64 = 8.0;

pub const SIGMOID_LUT: [u16; 256] = [
    1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4,
    4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 10,
    10, 11, 12, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 26,
    28, 30, 31, 33, 36, 38, 40, 43, 46, 49, 52, 55, 58, 62, 66, 70,
    75, 80, 85, 90, 96, 102, 108, 115, 122, 130, 138, 147, 156, 165, 176, 186,
    198, 210, 223, 237, 251, 266, 282, 299, 317, 336, 356, 377, 399, 422, 446, 472,
    498, 527, 556, 587, 619, 653, 688, 725, 763, 803, 844, 887, 931, 977, 1024, 1073,
    1124, 1176, 1229, 1284, 1340, 1397, 1455, 1514, 1575, 1636, 1698, 1761, 1824, 1888, 1952, 2016,
    2080, 2144, 2208, 2272, 2335, 2398, 2460, 2521, 2582, 2641, 2699, 2756, 2812, 2867, 2920, 2972,
    3023, 3072, 3119, 3165, 3209, 3252, 3293, 3333, 3371, 3408, 3443, 3477, 3509, 3540, 3569, 3598,
    3624, 3650, 3674, 3697, 3719, 3740, 3760, 3779, 3797, 3814, 3830, 3845, 3859, 3873, 3886, 3898,
    3910, 3920, 3931, 3940, 3949, 3958, 3966, 3974, 3981, 3988, 3994, 4000, 4006, 4011, 4016, 4021,
    4026, 4030, 4034, 4038, 4041, 4044, 4047, 4050, 4053, 4056, 4058, 4060, 4063, 4065, 4066, 4068,
    4070, 4072, 4073, 4074, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4084, 4084, 4085, 4086,
    4086, 4087, 4088, 4088, 4089, 4089, 4089, 4090, 4090, 4091, 4091, 4091, 4091, 4092, 4092, 4092,
    4092, 4093, 4093, 4093, 4093, 4093, 4094, 4094, 4094, 4094, 4094, 4094, 4094, 4094, 4095, 4095,
];

// === Isotonic Calibration LUT ===
pub const CALIBRATION_LUT_SIZE: usize = 64;

/// Input breakpoints (quantized, divide by 4096)
pub const CALIBRATION_X_Q: [u16; 64] = [
    0, 65, 130, 195, 260, 325, 390, 455,
    520, 585, 650, 715, 780, 845, 910, 975,
    1040, 1105, 1170, 1235, 1300, 1365, 1430, 1495,
    1560, 1625, 1690, 1755, 1820, 1885, 1950, 2015,
    2081, 2146, 2211, 2276, 2341, 2406, 2471, 2536,
    2601, 2666, 2731, 2796, 2861, 2926, 2991, 3056,
    3121, 3186, 3251, 3316, 3381, 3446, 3511, 3576,
    3641, 3706, 3771, 3836, 3901, 3966, 4031, 4096,
];

/// Output values (quantized, divide by 4096)
pub const CALIBRATION_Y_Q: [u16; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0,
    69, 69, 137, 137, 137, 137, 137, 137,
    137, 137, 137, 137, 137, 137, 137, 137,
    137, 137, 137, 137, 137, 137, 137, 292,
    354, 354, 354, 354, 354, 354, 354, 354,
    354, 529, 529, 896, 896, 896, 896, 1005,
    1005, 1005, 1005, 1005, 1005, 1186, 1186, 1186,
    1365, 2294, 2294, 2389, 2389, 2976, 3940, 4096,
];

// === Clinical Threshold ===
/// Optimal threshold for 98% recall
/// Use this instead of 0.5!
pub const CLINICAL_THRESHOLD: f64 = 0.03333333333333333;
pub const CLINICAL_THRESHOLD_Q: u16 = 137;
