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
    document.querySelectorAll("[data-filter]").forEach((row) => {
      const haystack = row.dataset.filter || "";
      row.classList.toggle("is-filtered", !haystack.includes(query));
    });
  });
}

function bindPlaylistPicker() {
  const checkboxes = Array.from(document.querySelectorAll("[data-playlist-checkbox]"));
  const categoryRows = Array.from(document.querySelectorAll("[data-category-accordion]"));
  const selectAll = document.querySelector("[data-select-all-playlists]");
  const counts = Array.from(document.querySelectorAll("[data-selection-count]"));
  const categoryCount = document.querySelector("[data-category-count]");
  const labelCount = document.querySelector("[data-label-count]");
  const labelCountInline = document.querySelector("[data-label-count-inline]");
  const submit = document.querySelector("[data-preview-submit]");
  const selectionFields = Array.from(document.querySelectorAll("[data-preview-selection-field]"));

  if (!checkboxes.length && !categoryRows.length) {
    return;
  }

  const labelsForRow = (row) => Array.from(row.querySelectorAll("[data-category-label]"));

  const setRowExpanded = (row, expanded) => {
    const panel = row.querySelector("[data-category-panel]");
    const toggle = row.querySelector("[data-category-toggle]");
    if (panel) {
      panel.classList.toggle("hidden", !expanded);
    }
    if (toggle) {
      toggle.setAttribute("aria-expanded", expanded ? "true" : "false");
    }
    if (expanded) {
      categoryRows.forEach((candidate) => {
        if (candidate !== row) {
          delete candidate.dataset.lastExpanded;
        }
      });
      row.dataset.lastExpanded = "true";
    } else {
      delete row.dataset.lastExpanded;
    }
  };

  const updateRowState = (row) => {
    const labels = labelsForRow(row);
    const selected = labels.filter((input) => input.checked).length;
    const master = row.querySelector("[data-category-master]");
    const allToggle = row.querySelector("[data-category-all]");
    const summary = row.querySelector("[data-category-summary]");

    row.classList.toggle("is-selected", selected > 0);
    if (master) {
      master.checked = selected > 0;
      master.indeterminate = selected > 0 && selected < labels.length;
    }
    if (allToggle) {
      allToggle.checked = labels.length > 0 && selected === labels.length;
      allToggle.indeterminate = selected > 0 && selected < labels.length;
    }
    if (summary) {
      if (selected === 0) {
        summary.textContent = "No labels selected";
      } else if (selected === labels.length) {
        summary.textContent = "All labels";
      } else {
        summary.textContent = `${selected} of ${labels.length} labels`;
      }
    }
  };

  const selectedCategoryCount = () =>
    categoryRows.filter((row) => labelsForRow(row).some((input) => input.checked)).length;

  const selectedLabelCount = () =>
    categoryRows.reduce((total, row) => total + labelsForRow(row).filter((input) => input.checked).length, 0);

  const collectSelectionState = () => {
    const categoryLabels = {};
    categoryRows.forEach((row) => {
      const categoryId = row.dataset.categoryId || "";
      const selectedLabels = labelsForRow(row)
        .filter((input) => input.checked)
        .map((input) => input.value);
      if (categoryId && selectedLabels.length) {
        categoryLabels[categoryId] = selectedLabels;
      }
    });
    const expandedRows = categoryRows.filter((row) => {
      const panel = row.querySelector("[data-category-panel]");
      return panel && !panel.classList.contains("hidden");
    });
    const expandedRow = expandedRows.find((row) => row.dataset.lastExpanded === "true") || expandedRows[0];
    return {
      selected_playlist_ids: checkboxes.filter((checkbox) => checkbox.checked).map((checkbox) => checkbox.value),
      category_labels: categoryLabels,
      expanded_category_id: expandedRow?.dataset.categoryId || "",
    };
  };

  const updateSelectionFields = () => {
    if (!selectionFields.length) {
      return;
    }
    const payload = JSON.stringify(collectSelectionState());
    selectionFields.forEach((field) => {
      field.value = payload;
    });
  };

  const update = () => {
    const selectedCount = checkboxes.filter((checkbox) => checkbox.checked).length;
    const categoryTotal = selectedCategoryCount();
    const labelTotal = selectedLabelCount();
    categoryRows.forEach(updateRowState);
    counts.forEach((count) => {
      count.textContent = String(selectedCount);
    });
    if (categoryCount) {
      categoryCount.textContent = String(categoryTotal);
    }
    if (labelCount) {
      labelCount.textContent = String(labelTotal);
    }
    if (labelCountInline) {
      labelCountInline.textContent = String(labelTotal);
    }
    if (submit) {
      submit.disabled = selectedCount === 0 || labelTotal === 0;
    }
    if (selectAll) {
      selectAll.textContent = selectedCount === checkboxes.length ? "Clear selection" : "Select all playlists";
    }
    updateSelectionFields();
  };

  checkboxes.forEach((checkbox) => {
    checkbox.addEventListener("change", update);
  });

  categoryRows.forEach((row) => {
    const master = row.querySelector("[data-category-master]");
    const allToggle = row.querySelector("[data-category-all]");
    const toggle = row.querySelector("[data-category-toggle]");
    const header = row.querySelector(".category-row-header");
    const labels = labelsForRow(row);

    if (master) {
      master.addEventListener("change", () => {
        labels.forEach((input) => {
          input.checked = master.checked;
        });
        if (master.checked) {
          setRowExpanded(row, true);
        }
        update();
      });
    }

    if (allToggle) {
      allToggle.addEventListener("change", () => {
        labels.forEach((input) => {
          input.checked = allToggle.checked;
        });
        update();
      });
    }

    if (toggle) {
      toggle.addEventListener("click", () => {
        const panel = row.querySelector("[data-category-panel]");
        const expanded = panel ? panel.classList.contains("hidden") : true;
        setRowExpanded(row, expanded);
        updateSelectionFields();
      });
    }

    if (header && toggle) {
      header.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof Element)) {
          return;
        }
        if (target.closest("input") || target.closest("button") || target.closest("label")) {
          return;
        }
        toggle.click();
      });
    }

    labels.forEach((input) => {
      input.addEventListener("change", update);
    });
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

  selectionFields.forEach((field) => {
    const form = field.closest("form");
    if (form) {
      form.addEventListener("submit", updateSelectionFields);
    }
  });

  update();
}

document.addEventListener("DOMContentLoaded", () => {
  bindLoadingForms();
  bindProgressForms();
  bindPlaylistToggle();
  bindRowFilter();
  bindPlaylistPicker();
});
