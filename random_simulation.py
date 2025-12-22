import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =============
# INPUT DATA
# =============
S1 = np.array([46646626, 39764246, 35948395, 32796034, 30371766,
               30785530, 30061336, 28880904, 28351248])   # observed chromosome sizes - R. cephalotes

S2 = np.array([84563487, 70579028, 68785893, 64095322, 62307233])  # ancestral genome - R. radicans

N_SIM = 100000 # simulation times
FUSION_RATE = 0.2 # fusion rate in the case of fissions
MIN_FRAGMENT = 1000000  # 1 Mbp


# ========================
# SIMULATION FUNCTIONS
# ========================
    
def draw_fusion_events(n_events, p=FUSION_RATE):
    """Draw number of fusion events using binomial model"""
    return np.random.binomial(n_events, p)


def random_break(chrom_length, n_pieces, min_frag=MIN_FRAGMENT):
    """
    Randomly break chromosome into n_pieces
    enforcing minimum fragment size.
    """
    if chrom_length < n_pieces * min_frag:
        raise ValueError("Chromosome too small for requested number of fragments")

    while True:
        breakpoints = sorted(
            random.randint(min_frag, chrom_length - min_frag)
            for _ in range(n_pieces - 1)
        )

        points = [0] + breakpoints + [chrom_length]
        fragments = [points[i+1] - points[i] for i in range(len(points)-1)]

        if min(fragments) >= min_frag:
            return fragments


def simulate_fission(S1, S2):
    """
    Fission simulation:
    Used when target species has more chromosomes than ancestral species
    Note that fission simulation allows fusion events as well
    """
    target_n = len(S1)
    ancestral_n = len(S2)
    target_len = int(sum(S1))

    # Number of event opportunities
    event_opportunities = target_n - ancestral_n - 1
    event_opportunities = max(event_opportunities, 0)

    # Draw number of fusion events
    n_fusions = draw_fusion_events(event_opportunities, p=FUSION_RATE)

    # Required number of fissions
    n_fissions = (target_n - ancestral_n) + n_fusions

    # Randomly assign fissions to ancestral chromosomes
    fission_targets = np.random.choice(
        range(len(S2)),
        size=n_fissions,
        replace=True
    )

    # Count how many fissions per ancestral chromosome
    fission_counts = np.bincount(fission_targets, minlength=len(S2))

    fragments = []

    for chrom, n_fis in zip(S2, fission_counts):
        n_pieces = 1 + n_fis
        fragments.extend(
            random_break(int(chrom), n_pieces, MIN_FRAGMENT)
        )


    # Step 2: Fusion events
    for _ in range(n_fusions):
        if len(fragments) >= 2:
            i, j = random.sample(range(len(fragments)), 2)
            fragments[i] += fragments[j]
            fragments.pop(j)

    # Step 3: Adjust total genome length
    current_len = sum(fragments)

    if current_len > target_len:
        # loss
        while sum(fragments) > target_len:
            idx = random.randrange(len(fragments))
            loss = min(fragments[idx], sum(fragments) - target_len)
            fragments[idx] -= loss
            if fragments[idx] <= 0:
                fragments.pop(idx)
                
        while sum(fragments) > target_len:
            idx = random.randrange(len(fragments))
            excess = sum(fragments) - target_len
            removable = fragments[idx] - MIN_FRAGMENT
            if removable <= 0:
                continue
            loss = min(removable, excess)
            fragments[idx] -= loss

    elif current_len < target_len:
        # duplication
        while sum(fragments) < target_len:
            fragments.append(random.choice(fragments))

    # Step 4: force correct chromosome count
    while len(fragments) > target_n:
        i, j = random.sample(range(len(fragments)), 2)
        fragments[i] += fragments[j]
        fragments.pop(j)

    while len(fragments) < target_n:
        idx = random.randrange(len(fragments))
        f = fragments[idx]
        if f < 2 * MIN_FRAGMENT:
            continue
        cut = random.randint(MIN_FRAGMENT, f - MIN_FRAGMENT)
        fragments[idx] = cut
        fragments.append(f - cut)

    return np.array(fragments)


def simulate_fusion(S1, S2):
    """
    Fusion-only simulation:
    Used when target species has fewer chromosomes than ancestral species
    """
    target_n = len(S1)
    target_len = int(sum(S1))

    # Start from ancestral chromosomes
    fragments = list(map(int, S2.copy()))

    # Step 1: Random fusion until chromosome number matches target
    while len(fragments) > target_n:
        i, j = random.sample(range(len(fragments)), 2)
        fragments[i] += fragments[j]
        fragments.pop(j)

    # Step 2: Adjust total genome length (only ADD length)
    current_len = sum(fragments)
    if current_len < target_len:
        while sum(fragments) < target_len:
            idx = random.randrange(len(fragments))
            fragments[idx] += 1  # add 1 bp at a time (integer-safe)

    elif current_len > target_len:
        # very rare biologically, but safe-guard
        while sum(fragments) > target_len:
            idx = random.randrange(len(fragments))
            removable = fragments[idx] - MIN_FRAGMENT
            if removable <= 0:
                continue
            fragments[idx] -= min(removable, sum(fragments) - target_len)

    return np.array(fragments)



# =========================
# RUN SIMULATION
# =========================
simulated_sizes = []
cvcl_sim = []

# Decide simulation mode
if len(S1) > len(S2):
    print("Running FISSION-dominated simulation")
    sim_func = simulate_fission
else:
    print("Running FUSION-only simulation")
    sim_func = simulate_fusion

for _ in range(N_SIM):
    sim = sim_func(S1, S2)
    simulated_sizes.append(sim)

    mean_len = np.mean(sim)
    sd_len = np.std(sim, ddof=1)
    cvcl_sim.append(sd_len / mean_len)

cvcl_sim = np.array(cvcl_sim)

# real CVCL
mean_obs = np.mean(S1)
sd_obs = np.std(S1, ddof=1)
cvcl_obs = sd_obs / mean_obs


# Calculate Z-score
mean_sim = np.mean(cvcl_sim)
sd_sim = np.std(cvcl_sim, ddof=1)

z_score = (cvcl_obs - mean_sim) / sd_sim

print("Observed CVCL:", cvcl_obs)
print("Mean simulated CVCL:", mean_sim)
print("SD simulated CVCL:", sd_sim)
print("Z-score of observed CVCL:", z_score)


with PdfPages("chromosome_breakage_simulation.pdf") as pdf:

    # ---------- Plot 2: CVCL distribution ----------
    plt.figure(figsize=(8,5))

    plt.hist(cvcl_sim, bins=100, color="steelblue", alpha=0.8,
             label="Simulated CVCL")
             
    plt.axvline(mean_sim, color="black", linestyle="--", linewidth=2,
            label="Mean simulated CVCL")
    plt.axvline(cvcl_obs, color="orange", linewidth=2,
                label="Observed CVCL")

    plt.xlabel("Coefficient of Variation of Chromosome Length (CVCL)")
    plt.ylabel("Frequency")
    plt.title("Distribution of CVCL from random ancestral breakage")
    plt.legend()
    plt.tight_layout()

    pdf.savefig()
    plt.close()



plt.hist(cvcl_sim, bins=100, color="steelblue", alpha=0.8,
         label="Simulated CVCL")


