import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")

# Read data
df = pd.read_csv("/Users/willroper/Downloads/MUSIC2_mem_test/Done.csv")

print(df)
LPTs = df["LPT"].values
Ls = df["L"].values[LPTs == 3]
Ns = df["N"].values[LPTs == 3]
mem = df["MaxRSS (GB)"].values[LPTs == 3]

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

ax.semilogy()

for L in [300, 600, 1000]:
    ax.plot(Ns[Ls == L][mem[Ls == L] != 0], mem[Ls == L][mem[Ls == L] != 0],
            label=str(L) + " cMpc")

ax.set_xlabel("$N_{\mathrm{part}}$")
ax.set_ylabel("MaxRSS (GB)")

new_ticks = np.array([128, 256, 512, 1024])
new_tick_labels = [str(i) for i in new_ticks]
ax.set_xticks(new_ticks)
ax.set_xticklabels(new_tick_labels)

ax.legend()

fig.savefig("plots/monofonic_memusage.png", bbox_inches="tight")


# Read data
df = pd.read_csv("/Users/willroper/Downloads/MUSIC2_mem_test/MPI_DMO.csv")

print(df)

LPTs = df["LPT"].values
Ns = df["N"].values
mem = df["MaxRSS x N_rank (GB)"].values
panph = df["Panphasia?"].values

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
axtwin = ax.twinx()

ax.semilogy()
axtwin.semilogy()

axtwin.grid(False)

for LPT in [2, 3]:
    okinds = np.logical_and(panph != "Y", np.logical_and(LPTs == LPT, mem != 0))
    ax.plot(Ns[okinds],
            mem[okinds],
            label="DMO LPT order = " + str(LPT))
    axtwin.plot(Ns[okinds],
                mem[okinds],
                alpha=0)

for LPT in [3]:
    okinds = np.logical_and(panph == "Y", np.logical_and(LPTs == LPT, mem != 0))
    ax.plot(Ns[okinds],
            mem[okinds],
            label="DMO + Panphasia LPT order = " + str(LPT), linestyle="--")
    axtwin.plot(Ns[okinds],
                mem[okinds],
                alpha=0)

# Read data
df = pd.read_csv("/Users/willroper/Downloads/MUSIC2_mem_test/MPI_Baryons.csv")

print(df)

Ns = df["N"].values
mem = df["MaxRSS x N_rank (GB)"].values

ax.plot(Ns[mem != 0],
        mem[mem != 0],
        label="Baryons LPT order = 2")

ax.set_xlabel("$N_{\mathrm{part}}$")
ax.set_ylabel(r"MaxRSS $\times N_{\mathrm{rank}}$ (GB) ")

new_ticks = np.array([128, 256, 512, 1024, 2048, 4096])
new_tick_labels = [str(i) for i in new_ticks]
ax.set_xticks(new_ticks)
ax.set_xticklabels(new_tick_labels)

new_tick_locations = np.array([i * 512 for i in [0.1, 0.2, 0.5, 1, 3, 6, 10, 17]])

def tick_function(x):
    return ["%.1f" % (im / 512) for im in x]

axtwin.set_yticks(new_tick_locations)
axtwin.set_yticklabels(tick_function(new_tick_locations))
axtwin.set_ylabel(r"$N_{\mathrm{Nodes}}$ (COSMA7)")

ax.legend()

fig.savefig("plots/monofonic_memusage_MPI.png", bbox_inches="tight")
