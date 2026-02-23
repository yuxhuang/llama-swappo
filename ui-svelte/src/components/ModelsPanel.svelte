<script lang="ts">
  import { models, loadModel, unloadAllModels, unloadSingleModel } from "../stores/api";
  import { isNarrow } from "../stores/theme";
  import { persistentStore } from "../stores/persistent";
  import type { Model } from "../lib/types";

  let isUnloading = $state(false);
  let menuOpen = $state(false);

  const showUnlistedStore = persistentStore<boolean>("showUnlisted", true);
  const showIdorNameStore = persistentStore<"id" | "name">("showIdorName", "id");

  let quantFilter = $state<string[]>([]);
  let capFilter = $state<string[]>([]);

  let availableQuants = $derived.by(() => {
    const quants = new Set<string>();
    $models.forEach((m) => {
      if (m.quantization_level && m.quantization_level !== "unknown") {
        quants.add(m.quantization_level);
      }
    });
    return Array.from(quants).sort();
  });

  let availableCaps = $derived.by(() => {
    const caps = new Set<string>();
    $models.forEach((m) => {
      m.capabilities?.forEach((c) => caps.add(c));
    });
    return Array.from(caps).sort();
  });

  let filteredModels = $derived.by(() => {
    const filtered = $models.filter((model) => {
      const matchesUnlisted = $showUnlistedStore || !model.unlisted;
      const matchesQuant = quantFilter.length === 0 || (model.quantization_level && quantFilter.includes(model.quantization_level));
      const matchesCap = capFilter.length === 0 || (model.capabilities && capFilter.some((c) => model.capabilities?.includes(c)));
      return matchesUnlisted && matchesQuant && matchesCap;
    });
    const peerModels = filtered.filter((m) => m.peerID);

    // Group peer models by peerID
    const grouped = peerModels.reduce(
      (acc, model) => {
        const peerId = model.peerID || "unknown";
        if (!acc[peerId]) acc[peerId] = [];
        acc[peerId].push(model);
        return acc;
      },
      {} as Record<string, Model[]>
    );

    return {
      regularModels: filtered.filter((m) => !m.peerID),
      peerModelsByPeerId: grouped,
    };
  });

  async function handleUnloadAllModels(): Promise<void> {
    isUnloading = true;
    try {
      await unloadAllModels();
    } catch (e) {
      console.error(e);
    } finally {
      setTimeout(() => (isUnloading = false), 1000);
    }
  }

  function toggleIdorName(): void {
    showIdorNameStore.update((prev) => (prev === "name" ? "id" : "name"));
  }

  function toggleShowUnlisted(): void {
    showUnlistedStore.update((prev) => !prev);
  }

  function getModelDisplay(model: Model): string {
    return $showIdorNameStore === "id" ? model.id : (model.name || model.id);
  }

  function toggleQuant(q: string): void {
    if (quantFilter.includes(q)) {
      quantFilter = quantFilter.filter((item) => item !== q);
    } else {
      quantFilter = [...quantFilter, q];
    }
  }

  function toggleCap(c: string): void {
    if (capFilter.includes(c)) {
      capFilter = capFilter.filter((item) => item !== c);
    } else {
      capFilter = [...capFilter, c];
    }
  }

  function getQuantColor(quant: string | undefined): string {
    if (!quant || quant === "unknown") return "bg-gray-100 dark:bg-white/10 text-gray-600 dark:text-gray-400 border-gray-200 dark:border-white/5";
    
    const q = quant.toUpperCase();
    if (q.includes("F32") || q.includes("F16") || q.includes("BF16")) {
      // Best: Pastel Green
      return "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800/30";
    }
    if (q.includes("Q8") || q.includes("IQ8")) {
      // Better: Pastel Teal
      return "bg-teal-100 dark:bg-teal-900/30 text-teal-700 dark:text-teal-400 border-teal-200 dark:border-teal-800/30";
    }
    if (q.includes("Q6") || q.includes("IQ6")) {
      // Great: Pastel Blue
      return "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-800/30";
    }
    if (q.includes("Q5") || q.includes("IQ5") || q.includes("Q4") || q.includes("IQ4") || q.includes("FP4")) {
      // Good: Pastel Yellow/Amber
      return "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 border-amber-200 dark:border-amber-800/30";
    }
    if (q.includes("Q3") || q.includes("IQ3") || q.includes("Q2") || q.includes("IQ2")) {
      // Okay: Pastel Orange/Grey
      return "bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400 border-orange-200 dark:border-orange-800/30";
    }
    
    return "bg-gray-100 dark:bg-white/10 text-gray-600 dark:text-gray-400 border-gray-200 dark:border-white/5";
  }

  function getSizeColor(model: Model): string {
    if (!model.parameter_size || model.parameter_size === "unknown") {
      return "bg-gray-100 dark:bg-white/10 text-gray-600 dark:text-gray-400 border-gray-200 dark:border-white/5";
    }

    // Parse parameters count (e.g., "7B" -> 7, "1.5B" -> 1.5, "1T" -> 1000)
    const paramsMatch = model.parameter_size.match(/(\d+(?:\.\d+)?)/);
    if (!paramsMatch) return "bg-gray-100 dark:bg-white/10 text-gray-600 dark:text-gray-400 border-gray-200 dark:border-white/5";
    
    let params = parseFloat(paramsMatch[1]);
    if (model.parameter_size.toUpperCase().includes("T")) {
      params *= 1000;
    } else if (model.parameter_size.toUpperCase().includes("M")) {
      params /= 1000;
    } else if (model.parameter_size.toUpperCase().includes("K")) {
      params /= 1000000;
    }
    
    // Bits per weight estimate
    let bits = 4; // default to 4-bit
    const q = (model.quantization_level || "").toUpperCase();
    
    if (q.includes("F32")) bits = 32;
    else if (q.includes("F16") || q.includes("BF16")) bits = 16;
    else if (q.includes("Q8") || q.includes("IQ8")) bits = 8;
    else if (q.includes("Q6") || q.includes("IQ6")) bits = 6;
    else if (q.includes("Q5") || q.includes("IQ5")) bits = 5;
    else if (q.includes("Q4") || q.includes("IQ4")) bits = 4;
    else if (q.includes("Q3") || q.includes("IQ3")) bits = 3;
    else if (q.includes("Q2") || q.includes("IQ2")) bits = 2;

    const estimatedGB = (params * bits) / 8;

    // Thresholds for 96GB VRAM
    // Very Fast: < 1/4 (24GB)
    // Fast: < 3/4 (72GB)
    // Slow: > 3/4 (72GB) or > 300B
    
    if (params > 300 || estimatedGB > 72) {
      // Slow: Pastel Red/Rose
      return "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border-red-200 dark:border-red-800/30";
    }
    
    if (estimatedGB < 24) {
      // Very Fast: Pastel Green
      return "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800/30";
    }
    
    // Fast: Pastel Blue
    return "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-800/30";
  }

  function getCapabilityColor(cap: string): string {
    const c = cap.toLowerCase();
    if (c === "tools") return "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 border-purple-200 dark:border-purple-800/30";
    if (c === "vision") return "bg-pink-100 dark:bg-pink-900/30 text-pink-700 dark:text-pink-400 border-pink-200 dark:border-pink-800/30";
    if (c === "embeddings") return "bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400 border-indigo-200 dark:border-indigo-800/30";
    if (c === "completion") return "bg-slate-100 dark:bg-white/10 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-white/5";
    return "bg-gray-100 dark:bg-white/10 text-gray-600 dark:text-gray-400 border-gray-200 dark:border-white/5";
  }
</script>

<div class="card h-full flex flex-col">
  <div class="shrink-0">
    <div class="flex justify-between items-baseline">
      <h2 class={$isNarrow ? "text-xl" : ""}>Models</h2>
      {#if $isNarrow}
        <div class="relative">
          <button class="btn text-base flex items-center gap-2 py-1" onclick={() => (menuOpen = !menuOpen)} aria-label="Toggle menu">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
              <path fill-rule="evenodd" d="M3 6.75A.75.75 0 0 1 3.75 6h16.5a.75.75 0 0 1 0 1.5H3.75A.75.75 0 0 1 3 6.75ZM3 12a.75.75 0 0 1 .75-.75h16.5a.75.75 0 0 1 0 1.5H3.75A.75.75 0 0 1 3 12Zm0 5.25a.75.75 0 0 1 .75-.75h16.5a.75.75 0 0 1 0 1.5H3.75a.75.75 0 0 1-.75-.75Z" clip-rule="evenodd" />
            </svg>
          </button>
          {#if menuOpen}
            <div class="absolute right-0 mt-2 w-48 bg-surface border border-gray-200 dark:border-white/10 rounded shadow-lg z-20">
              <button
                class="w-full text-left px-4 py-2 hover:bg-secondary-hover flex items-center gap-2"
                onclick={() => { toggleIdorName(); menuOpen = false; }}
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                  <path fill-rule="evenodd" d="M15.97 2.47a.75.75 0 0 1 1.06 0l4.5 4.5a.75.75 0 0 1 0 1.06l-4.5 4.5a.75.75 0 1 1-1.06-1.06l3.22-3.22H7.5a.75.75 0 0 1 0-1.5h11.69l-3.22-3.22a.75.75 0 0 1 0-1.06Zm-7.94 9a.75.75 0 0 1 0 1.06l-3.22 3.22H16.5a.75.75 0 0 1 0 1.5H4.81l3.22 3.22a.75.75 0 1 1-1.06 1.06l-4.5-4.5a.75.75 0 0 1 0-1.06l4.5-4.5a.75.75 0 0 1 1.06 0Z" clip-rule="evenodd" />
                </svg>
                {$showIdorNameStore === "id" ? "Show Name" : "Show ID"}
              </button>
              <button
                class="w-full text-left px-4 py-2 hover:bg-secondary-hover flex items-center gap-2"
                onclick={() => { toggleShowUnlisted(); menuOpen = false; }}
              >
                {#if $showUnlistedStore}
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                    <path d="M3.53 2.47a.75.75 0 0 0-1.06 1.06l18 18a.75.75 0 1 0 1.06-1.06l-18-18ZM22.676 12.553a11.249 11.249 0 0 1-2.631 4.31l-3.099-3.099a5.25 5.25 0 0 0-6.71-6.71L7.759 4.577a11.217 11.217 0 0 1 4.242-.827c4.97 0 9.185 3.223 10.675 7.69.12.362.12.752 0 1.113Z" />
                    <path d="M15.75 12c0 .18-.013.357-.037.53l-4.244-4.243A3.75 3.75 0 0 1 15.75 12ZM12.53 15.713l-4.243-4.244a3.75 3.75 0 0 0 4.244 4.243Z" />
                    <path d="M6.75 12c0-.619.107-1.213.304-1.764l-3.1-3.1a11.25 11.25 0 0 0-2.63 4.31c-.12.362-.12.752 0 1.114 1.489 4.467 5.704 7.69 10.675 7.69 1.5 0 2.933-.294 4.242-.827l-2.477-2.477A5.25 5.25 0 0 1 6.75 12Z" />
                  </svg>
                {:else}
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                    <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
                    <path fill-rule="evenodd" d="M1.323 11.447C2.811 6.976 7.028 3.75 12.001 3.75c4.97 0 9.185 3.223 10.675 7.69.12.362.12.752 0 1.113-1.487 4.471-5.705 7.697-10.677 7.697-4.97 0-9.186-3.223-10.675-7.69a1.762 1.762 0 0 1 0-1.113ZM17.25 12a5.25 5.25 0 1 1-10.5 0 5.25 5.25 0 0 1 10.5 0Z" clip-rule="evenodd" />
                  </svg>
                {/if}
                {$showUnlistedStore ? "Hide Unlisted" : "Show Unlisted"}
              </button>
              <button
                class="w-full text-left px-4 py-2 hover:bg-secondary-hover flex items-center gap-2"
                onclick={() => { handleUnloadAllModels(); menuOpen = false; }}
                disabled={isUnloading}
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-6 h-6">
                  <path fill-rule="evenodd" d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25Zm.53 5.47a.75.75 0 0 0-1.06 0l-3 3a.75.75 0 1 0 1.06 1.06l1.72-1.72v5.69a.75.75 0 0 0 1.5 0v-5.69l1.72 1.72a.75.75 0 1 0 1.06-1.06l-3-3Z" clip-rule="evenodd" />
                </svg>
                {isUnloading ? "Unloading..." : "Unload All"}
              </button>
            </div>
          {/if}
        </div>
      {/if}
    </div>
    {#if !$isNarrow}
      <div class="flex justify-between">
        <div class="flex gap-2">
          <button class="btn text-base flex items-center gap-2" onclick={toggleIdorName} style="line-height: 1.2">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
              <path fill-rule="evenodd" d="M15.97 2.47a.75.75 0 0 1 1.06 0l4.5 4.5a.75.75 0 0 1 0 1.06l-4.5 4.5a.75.75 0 1 1-1.06-1.06l3.22-3.22H7.5a.75.75 0 0 1 0-1.5h11.69l-3.22-3.22a.75.75 0 0 1 0-1.06Zm-7.94 9a.75.75 0 0 1 0 1.06l-3.22 3.22H16.5a.75.75 0 0 1 0 1.5H4.81l3.22 3.22a.75.75 0 1 1-1.06 1.06l-4.5-4.5a.75.75 0 0 1 0-1.06l4.5-4.5a.75.75 0 0 1 1.06 0Z" clip-rule="evenodd" />
            </svg>
            {$showIdorNameStore === "id" ? "ID" : "Name"}
          </button>

          <button class="btn text-base flex items-center gap-2" onclick={toggleShowUnlisted} style="line-height: 1.2">
            {#if $showUnlistedStore}
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                <path d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
                <path fill-rule="evenodd" d="M1.323 11.447C2.811 6.976 7.028 3.75 12.001 3.75c4.97 0 9.185 3.223 10.675 7.69.12.362.12.752 0 1.113-1.487 4.471-5.705 7.697-10.677 7.697-4.97 0-9.186-3.223-10.675-7.69a1.762 1.762 0 0 1 0-1.113ZM17.25 12a5.25 5.25 0 1 1-10.5 0 5.25 5.25 0 0 1 10.5 0Z" clip-rule="evenodd" />
              </svg>
            {:else}
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                <path d="M3.53 2.47a.75.75 0 0 0-1.06 1.06l18 18a.75.75 0 1 0 1.06-1.06l-18-18ZM22.676 12.553a11.249 11.249 0 0 1-2.631 4.31l-3.099-3.099a5.25 5.25 0 0 0-6.71-6.71L7.759 4.577a11.217 11.217 0 0 1 4.242-.827c4.97 0 9.185 3.223 10.675 7.69.12.362.12.752 0 1.113Z" />
                <path d="M15.75 12c0 .18-.013.357-.037.53l-4.244-4.243A3.75 3.75 0 0 1 15.75 12ZM12.53 15.713l-4.243-4.244a3.75 3.75 0 0 0 4.244 4.243Z" />
                <path d="M6.75 12c0-.619.107-1.213.304-1.764l-3.1-3.1a11.25 11.25 0 0 0-2.63 4.31c-.12.362-.12.752 0 1.114 1.489 4.467 5.704 7.69 10.675 7.69 1.5 0 2.933-.294 4.242-.827l-2.477-2.477A5.25 5.25 0 0 1 6.75 12Z" />
              </svg>
            {/if}
            unlisted
          </button>
        </div>
        <button class="btn text-base flex items-center gap-2" onclick={handleUnloadAllModels} disabled={isUnloading}>
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-6 h-6">
            <path fill-rule="evenodd" d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25Zm.53 5.47a.75.75 0 0 0-1.06 0l-3 3a.75.75 0 1 0 1.06 1.06l1.72-1.72v5.69a.75.75 0 0 0 1.5 0v-5.69l1.72 1.72a.75.75 0 1 0 1.06-1.06l-3-3Z" clip-rule="evenodd" />
          </svg>
          {isUnloading ? "Unloading..." : "Unload All"}
        </button>
      </div>
    {/if}

    <div class="mt-4 flex flex-col gap-2 border-b border-gray-200 dark:border-white/10 pb-4">
      {#if availableQuants.length > 0}
        <div class="flex flex-wrap items-center gap-2">
          <span class="text-xs font-semibold text-txtsecondary min-w-[80px]">Quantization:</span>
          <div class="flex flex-wrap gap-1">
            {#each availableQuants as q}
              <button 
                class="filter-tag {quantFilter.includes(q) ? 'filter-tag--active' : ''}" 
                onclick={() => toggleQuant(q)}
              >
                {q}
              </button>
            {/each}
          </div>
        </div>
      {/if}
      {#if availableCaps.length > 0}
        <div class="flex flex-wrap items-center gap-2">
          <span class="text-xs font-semibold text-txtsecondary min-w-[80px]">Capabilities:</span>
          <div class="flex flex-wrap gap-1">
            {#each availableCaps as c}
              <button 
                class="filter-tag {capFilter.includes(c) ? 'filter-tag--active' : ''}" 
                onclick={() => toggleCap(c)}
              >
                {c}
              </button>
            {/each}
          </div>
        </div>
      {/if}
    </div>
  </div>

  <div class="flex-1 overflow-y-auto">
    <table class="w-full">
      <thead class="sticky top-0 bg-card z-10">
        <tr class="text-left border-b border-gray-200 dark:border-white/10 bg-surface">
          <th>{$showIdorNameStore === "id" ? "Model ID" : "Name"}</th>
          <th></th>
          <th>State</th>
        </tr>
      </thead>
      <tbody>
        {#each filteredModels.regularModels as model (model.id)}
          <tr class="border-b hover:bg-secondary-hover border-gray-200">
            <td class={model.unlisted ? "text-txtsecondary" : ""}>
              <div class="flex items-center gap-2">
                <a href="/upstream/{model.id}/" class="font-semibold" target="_blank">
                  {getModelDisplay(model)}
                </a>
                {#if model.quantization_level && model.quantization_level !== "unknown"}
                  <span class="px-1.5 py-0.5 text-[10px] font-bold rounded-md uppercase tracking-wider border shrink-0 {getQuantColor(model.quantization_level)}">
                    {model.quantization_level}
                  </span>
                {/if}
                {#if model.parameter_size && model.parameter_size !== "unknown"}
                  <span class="px-1.5 py-0.5 text-[10px] font-bold rounded-md uppercase tracking-wider border shrink-0 {getSizeColor(model)}">
                    {model.parameter_size}
                  </span>
                {/if}
              </div>
              {#if model.capabilities && model.capabilities.length > 0}
                <div class="flex flex-wrap gap-1 mt-1">
                  {#each model.capabilities as cap}
                    <span class="px-1.5 py-0.2 text-[9px] font-medium rounded-sm lowercase border shrink-0 {getCapabilityColor(cap)}">
                      {cap}
                    </span>
                  {/each}
                </div>
              {/if}
              {#if model.description}
                <p class={(model.unlisted ? "text-opacity-70" : "") + " ml-0"}><em>{model.description}</em></p>
              {/if}
            </td>
            <td class="w-12">
              {#if model.state === "stopped"}
                <button class="btn btn--sm" onclick={() => loadModel(model.id)}>Load</button>
              {:else}
                <button class="btn btn--sm" onclick={() => unloadSingleModel(model.id)} disabled={model.state !== "ready"}>Unload</button>
              {/if}
            </td>
            <td class="w-20">
              <span class="w-16 text-center status status--{model.state}">{model.state}</span>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>

    {#if Object.keys(filteredModels.peerModelsByPeerId).length > 0}
      <h3 class="mt-8 mb-2">Peer Models</h3>
      {#each Object.entries(filteredModels.peerModelsByPeerId).sort(([a], [b]) => a.localeCompare(b)) as [peerId, peerModels] (peerId)}
        <div class="mb-4">
          <table class="w-full">
            <thead class="sticky top-0 bg-card z-10">
              <tr class="text-left border-b border-gray-200 dark:border-white/10 bg-surface">
                <th class="font-semibold">{peerId}</th>
              </tr>
            </thead>
            <tbody>
              {#each peerModels as model (model.id)}
                <tr class="border-b hover:bg-secondary-hover border-gray-200">
                  <td class="pl-8 {model.unlisted ? 'text-txtsecondary' : ''}">
                    <span>{model.id}</span>
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  .filter-tag {
    display: inline-block;
    padding: 0.125rem 0.5rem;
    font-size: 10px;
    font-weight: 500;
    border-radius: 0.25rem;
    border: 1px solid #e5e7eb;
    background-color: #f9fafb;
    color: #4b5563;
    transition: all 0.2s;
    cursor: pointer;
  }

  :global(.dark) .filter-tag {
    border-color: rgba(255, 255, 255, 0.1);
    background-color: rgba(255, 255, 255, 0.05);
    color: #9ca3af;
  }

  .filter-tag:hover {
    background-color: #f3f4f6;
  }

  :global(.dark) .filter-tag:hover {
    background-color: rgba(255, 255, 255, 0.1);
  }

  .filter-tag--active {
    background-color: rgba(50, 184, 198, 0.1);
    border-color: rgba(50, 184, 198, 0.3);
    color: #32b8c6;
  }

  .filter-tag--active:hover {
    background-color: rgba(50, 184, 198, 0.2);
  }

  :global(.dark) .filter-tag--active {
    background-color: rgba(33, 128, 141, 0.2);
    border-color: rgba(33, 128, 141, 0.4);
    color: #21808d;
  }

  :global(.dark) .filter-tag--active:hover {
    background-color: rgba(33, 128, 141, 0.3);
  }
</style>
