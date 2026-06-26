function showLoadingOverlay(title, message, options = {}) {
  const overlay = document.getElementById("loading-overlay");
  const titleNode = document.getElementById("loading-title");
  const messageNode = document.getElementById("loading-message");
  const progressText = document.getElementById("loading-progress-text");
  const progressBar = document.getElementById("loading-progress-bar");

  if (!overlay || !titleNode || !messageNode) {
    return;
  }

  titleNode.textContent = title || "Working...";
  messageNode.textContent = message || "Please keep this tab open.";
  overlay.classList.toggle("is-progress", Boolean(options.progress));
  if (progressText) {
    progressText.textContent = options.progressText || "";
  }
  if (progressBar) {
    progressBar.style.width = options.progress ? `${options.percent || 0}%` : "";
  }
  overlay.classList.add("is-visible");
  overlay.setAttribute("aria-hidden", "false");
}

function updateLoadingProgress({ message, percent, current, total }) {
  const messageNode = document.getElementById("loading-message");
  const progressText = document.getElementById("loading-progress-text");
  const progressBar = document.getElementById("loading-progress-bar");
  const boundedPercent = Math.max(0, Math.min(100, Number(percent || 0)));

  if (messageNode && message) {
    messageNode.textContent = message;
  }
  if (progressBar) {
    progressBar.style.width = `${boundedPercent}%`;
  }
  if (progressText) {
    const stepText = current && total ? `Step ${current} of ${total}` : "Syncing";
    progressText.textContent = `${stepText} - ${boundedPercent}%`;
  }
}

function markFormSubmitting(form) {
  form.setAttribute("aria-busy", "true");
  form.classList.add("is-submitting");

  form.querySelectorAll("button, input:not([type='hidden']), textarea").forEach((control) => {
    control.setAttribute("readonly", "readonly");
    if (control.tagName === "BUTTON") {
      control.setAttribute("disabled", "disabled");
    }
  });
}

function bindLoadingForms() {
  document.querySelectorAll("form[data-loading-title]").forEach((form) => {
    if (form.dataset.progressForm !== undefined) {
      return;
    }
    form.addEventListener("submit", () => {
      showLoadingOverlay(form.dataset.loadingTitle, form.dataset.loadingMessage);
      markFormSubmitting(form);
    });
  });
}

async function pollApplyStatus(statusUrl) {
  while (true) {
    const response = await fetch(statusUrl, { headers: { Accept: "application/json" } });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Could not read apply progress.");
    }

    updateLoadingProgress(payload);

    if (payload.status === "complete") {
      window.location.href = payload.finish_url || "/finish";
      return;
    }

    if (payload.status === "failed") {
      throw new Error(payload.error || "Applying playlists failed.");
    }

    await new Promise((resolve) => setTimeout(resolve, 900));
  }
}

function bindProgressForms() {
  document.querySelectorAll("form[data-progress-form]").forEach((form) => {
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      showLoadingOverlay(form.dataset.loadingTitle, form.dataset.loadingMessage, {
        progress: true,
        progressText: "Starting - 0%",
        percent: 0,
      });
      markFormSubmitting(form);

      try {
        const response = await fetch(form.dataset.progressStartUrl || form.action, {
          method: "POST",
          body: new FormData(form),
          headers: { Accept: "application/json" },
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Could not start YouTube sync.");
        }
        await pollApplyStatus(payload.status_url);
      } catch (error) {
        updateLoadingProgress({
          message: error instanceof Error ? error.message : "Applying playlists failed.",
          percent: 100,
        });
      }
    });
  });
}

function bindPlaylistToggle() {
  const scope = document.querySelector("[data-playlist-toggle]");
  const wrap = document.getElementById("playlist-select-wrap");

  if (!scope || !wrap) {
    return;
  }

  const update = () => {
    wrap.classList.toggle("hidden", scope.value !== "single_playlist");
  };

  scope.addEventListener("change", update);
  update();
}

function bindRowFilter() {
  const input = document.querySelector("[data-row-filter]");

  if (!input) {
    return;
  }

  input.addEventListener("input", () => {
    const query = input.value.toLowerCase().trim();
    document.querySelectorAll("tbody tr[data-filter]").forEach((row) => {
      const haystack = row.dataset.filter || "";
      row.classList.toggle("is-filtered", !haystack.includes(query));
    });
  });
}

function bindPlaylistPicker() {
  const checkboxes = Array.from(document.querySelectorAll("[data-playlist-checkbox]"));
  const categoryCheckboxes = Array.from(document.querySelectorAll("[data-category-checkbox]"));
  const selectAll = document.querySelector("[data-select-all-playlists]");
  const count = document.querySelector("[data-selection-count]");
  const categoryCount = document.querySelector("[data-category-count]");
  const submit = document.querySelector("[data-preview-submit]");

  if (!checkboxes.length && !categoryCheckboxes.length) {
    return;
  }

  const update = () => {
    const selectedCount = checkboxes.filter((checkbox) => checkbox.checked).length;
    const selectedCategoryCount = categoryCheckboxes.filter((checkbox) => checkbox.checked).length;
    if (count) {
      count.textContent = String(selectedCount);
    }
    if (categoryCount) {
      categoryCount.textContent = String(selectedCategoryCount);
    }
    if (submit) {
      submit.disabled = selectedCount === 0 || selectedCategoryCount === 0;
    }
    if (selectAll) {
      selectAll.textContent = selectedCount === checkboxes.length ? "Clear selection" : "Select all playlists";
    }
  };

  checkboxes.forEach((checkbox) => {
    checkbox.addEventListener("change", update);
  });
  categoryCheckboxes.forEach((checkbox) => {
    checkbox.addEventListener("change", update);
  });

  if (selectAll) {
    selectAll.addEventListener("click", () => {
      const shouldSelectAll = checkboxes.some((checkbox) => !checkbox.checked);
      checkboxes.forEach((checkbox) => {
        checkbox.checked = shouldSelectAll;
      });
      update();
    });
  }

  update();
}

document.addEventListener("DOMContentLoaded", () => {
  bindLoadingForms();
  bindProgressForms();
  bindPlaylistToggle();
  bindRowFilter();
  bindPlaylistPicker();
});
