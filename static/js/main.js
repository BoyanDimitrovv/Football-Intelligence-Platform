/* ==========================================================================
   main.js - UI interactions for the Football Intelligence Platform
   ========================================================================== */

(function () {
  "use strict";

  /* ------------------------------------------------------- filter panel --- */
  function initFilters() {
    const form = document.getElementById("filter-form");
    if (!form) return;

    const output = document.getElementById("overall-output");
    const slider = document.getElementById("min_overall");
    if (output && slider) {
      const sync = () => {
        output.textContent = slider.value;
      };
      slider.addEventListener("input", sync);
      sync();
    }

    // Selects apply immediately; text inputs need an explicit submit.
    form.querySelectorAll("select").forEach((select) => {
      select.addEventListener("change", () => form.submit());
    });

    form.querySelectorAll('input[type="number"]').forEach((input) => {
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") form.submit();
      });
    });
  }

  /* -------------------------------------------------------- compare bar --- */
  function initCompareSelection() {
    const bar = document.getElementById("compare-bar");
    const checkboxes = Array.from(document.querySelectorAll(".compare-check"));
    if (!bar || !checkboxes.length) return;

    const countLabel = document.getElementById("compare-count");
    const namesLabel = document.getElementById("compare-names");
    const link = document.getElementById("compare-link");
    const clear = document.getElementById("compare-clear");
    const max = (window.PLAYER_COMPARE && window.PLAYER_COMPARE.max) || 4;

    function selected() {
      return checkboxes.filter((box) => box.checked);
    }

    function refresh() {
      const chosen = selected();
      countLabel.textContent = chosen.length;
      namesLabel.textContent = chosen
        .map((box) => box.closest(".player-card").querySelector(".name").textContent.trim())
        .join(" · ");
      link.href = `${window.PLAYER_COMPARE.compareUrl}?ids=${chosen.map((box) => box.value).join(",")}`;

      const full = chosen.length >= max;
      checkboxes.forEach((box) => {
        if (!box.checked && full) {
          box.disabled = true;
          box.closest(".player-card").style.opacity = "0.55";
        } else {
          box.disabled = false;
          box.closest(".player-card").style.opacity = "";
        }
      });

      bar.style.display = chosen.length ? "block" : "none";
    }

    checkboxes.forEach((box) => box.addEventListener("change", refresh));
    clear.addEventListener("click", () => {
      checkboxes.forEach((box) => { box.checked = false; });
      refresh();
    });
    refresh();
  }

  /* ------------------------------------------------------------- chat ----- */
  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function formatMessage(text) {
    let html = escapeHtml(text);
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    return html
      .split(/\n{2,}/)
      .map((block) => {
        const lines = block.split("\n");
        if (lines.length && lines.every((line) => line.trim().startsWith("•"))) {
          const items = lines
            .map((line) => `<li>${line.replace(/^\s*•\s*/, "")}</li>`)
            .join("");
          return `<ul>${items}</ul>`;
        }
        return `<p>${block.replace(/\n/g, "<br>")}</p>`;
      })
      .join("");
  }

  function appendMessage(container, role, html, cards) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "Вие" : "AI";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = html;

    wrapper.appendChild(avatar);
    wrapper.appendChild(bubble);

    if (cards && cards.length) {
      const cardRow = document.createElement("div");
      cardRow.className = "chat-cards";
      cards.forEach((card) => {
        const link = document.createElement("a");
        link.className = "chat-card";
        link.href = card.url || `/players/${card.id}`;
        link.innerHTML =
          `<span class="ov">${card.overall}</span>` +
          `<span>${card.name}<br><small style="color:var(--muted)">${card.position_label} · ${card.team}</small></span>`;
        cardRow.appendChild(link);
      });
      wrapper.appendChild(cardRow);
    }

    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
    return wrapper;
  }

  function showTyping(container) {
    const wrapper = document.createElement("div");
    wrapper.className = "message bot";
    wrapper.id = "typing-indicator";
    wrapper.innerHTML =
      '<div class="avatar">AI</div>' +
      '<div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div>';
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
  }

  function initChat() {
    const form = document.getElementById("chat-form");
    const messages = document.getElementById("chat-messages");
    const input = document.getElementById("chat-question");
    if (!form || !messages || !input) return;

    const chips = document.getElementById("suggestion-chips");
    const clear = document.getElementById("chat-clear");

    async function send(question) {
      const text = question.trim();
      if (!text) return;

      appendMessage(messages, "user", `<p>${escapeHtml(text)}</p>`);
      input.value = "";
      showTyping(messages);

      try {
        const response = await fetch("/api/assistant", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text }),
        });
        const data = await response.json();
        const indicator = document.getElementById("typing-indicator");
        if (indicator) indicator.remove();

        if (data.error) {
          appendMessage(messages, "bot", `<p>${escapeHtml(data.error)}</p>`);
          return;
        }
        appendMessage(messages, "bot", formatMessage(data.reply), data.cards);

        if (data.suggestions && data.suggestions.length && chips) {
          renderSuggestions(chips, data.suggestions);
        }
      } catch (error) {
        const indicator = document.getElementById("typing-indicator");
        if (indicator) indicator.remove();
        appendMessage(
          messages,
          "bot",
          "<p>Възникна проблем при връзката с асистента. Опитайте отново след момент.</p>"
        );
      }
    }

    function renderSuggestions(container, suggestions) {
      container.innerHTML = "";
      suggestions.forEach((suggestion) => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        chip.textContent = suggestion;
        chip.addEventListener("click", () => send(suggestion));
        container.appendChild(chip);
      });
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      send(input.value);
    });

    if (chips) {
      chips.addEventListener("click", (event) => {
        const chip = event.target.closest(".chip");
        if (!chip) return;
        send(chip.textContent);
      });
    }

    if (clear) {
      clear.addEventListener("click", () => {
        messages.innerHTML = "";
        appendMessage(
          messages,
          "bot",
          "<p>Нов разговор. Питайте за играчи, позиции, отбори или статистика.</p>"
        );
      });
    }
  }

  /* ------------------------------------------------------ attribute bars -- */
  function animateBars() {
    const bars = document.querySelectorAll(".attr-bar span");
    if (!bars.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          const bar = entry.target;
          const width = bar.style.width;
          bar.style.width = "0%";
          requestAnimationFrame(() => {
            bar.style.width = width;
          });
          observer.unobserve(bar);
        });
      },
      { threshold: 0.2 }
    );

    bars.forEach((bar) => observer.observe(bar));
  }

  document.addEventListener("DOMContentLoaded", () => {
    initFilters();
    initCompareSelection();
    initChat();
    animateBars();
  });
})();
