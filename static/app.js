function showLoadingOverlay(title, message) {
  const overlay = document.getElementById("loading-overlay");
  const titleNode = document.getElementById("loading-title");
  const messageNode = document.getElementById("loading-message");

  if (!overlay || !titleNode || !messageNode) {
    return;
  }

  titleNode.textContent = title || "Working...";
  messageNode.textContent = message || "Please keep this tab open.";
  overlay.classList.add("is-visible");
  overlay.setAttribute("aria-hidden", "false");
}

function bindLoadingForms() {
  document.querySelectorAll("form[data-loading-title]").forEach((form) => {
    form.addEventListener("submit", () => {
      showLoadingOverlay(form.dataset.loadingTitle, form.dataset.loadingMessage);
      form.setAttribute("aria-busy", "true");
      form.classList.add("is-submitting");

      form.querySelectorAll("button, input:not([type='hidden']), textarea").forEach((control) => {
        control.setAttribute("readonly", "readonly");
        if (control.tagName === "BUTTON") {
          control.setAttribute("disabled", "disabled");
        }
      });
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
  const selectAll = document.querySelector("[data-select-all-playlists]");
  const count = document.querySelector("[data-selection-count]");
  const submit = document.querySelector("[data-preview-submit]");

  if (!checkboxes.length) {
    return;
  }

  const update = () => {
    const selectedCount = checkboxes.filter((checkbox) => checkbox.checked).length;
    if (count) {
      count.textContent = String(selectedCount);
    }
    if (submit) {
      submit.disabled = selectedCount === 0;
    }
    if (selectAll) {
      selectAll.textContent = selectedCount === checkboxes.length ? "Clear selection" : "Select all playlists";
    }
  };

  checkboxes.forEach((checkbox) => {
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
  bindPlaylistToggle();
  bindRowFilter();
  bindPlaylistPicker();
});
