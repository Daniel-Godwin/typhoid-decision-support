/* Progressive enhancement only — every page works without JavaScript. */
(function () {
  "use strict";

  /* Mobile navigation ---------------------------------------------------- */
  var toggle = document.querySelector("[data-nav-toggle]");
  var scrim = document.querySelector("[data-scrim]");
  function closeNav() { document.body.classList.remove("nav-open"); if (toggle) toggle.setAttribute("aria-expanded", "false"); }
  if (toggle) {
    toggle.addEventListener("click", function () {
      var open = document.body.classList.toggle("nav-open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }
  if (scrim) scrim.addEventListener("click", closeNav);

  /* User menu ------------------------------------------------------------ */
  var trigger = document.querySelector("[data-menu-trigger]");
  var menu = document.querySelector("[data-menu]");
  if (trigger && menu) {
    trigger.addEventListener("click", function (e) {
      e.stopPropagation();
      var open = menu.hidden;
      menu.hidden = !open;
      trigger.setAttribute("aria-expanded", open ? "true" : "false");
    });
    document.addEventListener("click", function (e) {
      if (!menu.hidden && !menu.contains(e.target)) {
        menu.hidden = true;
        trigger.setAttribute("aria-expanded", "false");
      }
    });
  }

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") {
      closeNav();
      if (menu && !menu.hidden) { menu.hidden = true; if (trigger) trigger.setAttribute("aria-expanded", "false"); }
    }
  });

  /* Flash messages: dismissible, and auto-dismiss the non-critical ones --- */
  document.querySelectorAll("[data-flash]").forEach(function (flash) {
    var close = flash.querySelector("[data-dismiss]");
    function remove() {
      flash.style.transition = "opacity 180ms, transform 180ms";
      flash.style.opacity = "0";
      flash.style.transform = "translateX(12px)";
      setTimeout(function () { flash.remove(); }, 200);
    }
    if (close) close.addEventListener("click", remove);
    if (flash.classList.contains("success") || flash.classList.contains("info")) {
      setTimeout(remove, 6000);
    }
  });

  /* Assessment form: fill with population reference values ---------------- */
  var fillBtn = document.querySelector("[data-fill-reference]");
  if (fillBtn) {
    fillBtn.addEventListener("click", function () {
      var reference;
      try { reference = JSON.parse(fillBtn.getAttribute("data-reference") || "{}"); }
      catch (err) { return; }
      var form = document.getElementById("assessment-form");
      if (!form) return;
      Object.keys(reference).forEach(function (key) {
        var el = form.elements[key];
        if (el) { el.value = reference[key]; el.dispatchEvent(new Event("change", { bubbles: true })); }
      });
    });
  }

  /* Guard against double submission -------------------------------------- */
  document.querySelectorAll("form[data-guard]").forEach(function (form) {
    form.addEventListener("submit", function () {
      var button = form.querySelector('button[type="submit"]');
      if (!button || button.dataset.busy) return;
      button.dataset.busy = "1";
      button.setAttribute("aria-disabled", "true");
      var original = button.textContent;
      button.textContent = button.dataset.busyLabel || "Working…";
      setTimeout(function () {
        delete button.dataset.busy;
        button.removeAttribute("aria-disabled");
        button.textContent = original;
      }, 12000);
    });
  });

  /* Confirmation prompts for destructive or state-changing actions -------- */
  document.querySelectorAll("form[data-confirm]").forEach(function (form) {
    form.addEventListener("submit", function (e) {
      if (!window.confirm(form.getAttribute("data-confirm"))) e.preventDefault();
    });
  });

  /* Mark the current sidebar item ---------------------------------------- */
  var here = window.location.pathname;
  document.querySelectorAll(".nav a").forEach(function (link) {
    var href = link.getAttribute("href");
    if (!href || href === "#") return;
    if (here === href || (href !== "/" && here.indexOf(href) === 0)) link.classList.add("active");
  });
})();
