ETF_UNIVERSE: list[str] = [
    # US Broad Market
    "SPY", "IVV", "VOO", "VTI", "ITOT", "SCHB", "VV", "IWB",
    # Growth / Value
    "IWF", "IWD", "VUG", "VTV", "SCHG", "SCHV", "IVW", "IVE",
    # Small / Mid Cap
    "IWM", "IJH", "IJR", "MDY", "VO", "VB", "SCHA", "VXF",
    # Sector — SPDR
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLB", "XLRE", "XLU",
    # Sector — Vanguard
    "VGT", "VFH", "VDE", "VHT", "VIS", "VOX", "VCR", "VDC", "VAW", "VNQ", "VPU",
    # Tech / Growth Thematic
    "QQQ", "QQQM", "SMH", "SOXX", "ARKK", "ARKW", "ARKG", "ARKF", "BOTZ", "ROBO",
    # International Developed
    "VEA", "VXUS", "EFA", "IEFA", "SCHF", "VEU",
    # International Emerging
    "EEM", "VWO", "IEMG", "MCHI", "FXI", "KWEB",
    # Dividend
    "VIG", "VYM", "SCHD", "DVY", "HDV", "DGRO", "SDY",
    # Real Estate
    "VNQ", "SCHH", "IYR", "RWR",
    # Fixed Income (for completeness — equity holdings minimal)
    "AGG", "BND",
    # Commodities / Alternatives
    "GLD", "IAU", "SLV",
    # Clean Energy / ESG
    "ICLN", "LIT", "ESGU", "ESGV",
    # Multi-factor
    "QUAL", "MTUM", "VLUE", "SIZE", "USMV", "EFAV",
]

# Remove duplicates while preserving order
_seen: set[str] = set()
ETF_UNIVERSE = [e for e in ETF_UNIVERSE if not (e in _seen or _seen.add(e))]
