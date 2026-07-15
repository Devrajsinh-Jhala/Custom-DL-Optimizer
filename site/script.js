const header = document.querySelector("[data-header]");
const nav = document.querySelector("[data-nav]");
const navToggle = document.querySelector("[data-nav-toggle]");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const setHeaderState = () => {
  header?.classList.toggle("is-scrolled", window.scrollY > 24);
};

setHeaderState();
window.addEventListener("scroll", setHeaderState, { passive: true });

navToggle?.addEventListener("click", () => {
  const open = nav?.classList.toggle("is-open") ?? false;
  navToggle.setAttribute("aria-expanded", String(open));
  navToggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
});

nav?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    nav.classList.remove("is-open");
    navToggle?.setAttribute("aria-expanded", "false");
    navToggle?.setAttribute("aria-label", "Open navigation");
  });
});

const writeClipboard = async (value) => {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = value;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.append(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) throw new Error("Clipboard copy was rejected");
};

const copyText = async (value, button) => {
  const defaultLabel = button.getAttribute("aria-label") ?? "Copy";
  try {
    await writeClipboard(value);
    button.setAttribute("aria-label", "Copied");
    button.innerHTML = '<i data-lucide="check" aria-hidden="true"></i>';
    window.lucide?.createIcons();
    window.setTimeout(() => {
      button.setAttribute("aria-label", defaultLabel);
      button.innerHTML = '<i data-lucide="copy" aria-hidden="true"></i>';
      window.lucide?.createIcons();
    }, 1600);
  } catch {
    button.setAttribute("aria-label", "Copy failed");
  }
};

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", () => copyText(button.dataset.copy, button));
});

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", () => {
    const target = document.getElementById(button.dataset.copyTarget);
    copyText(target?.textContent ?? "", button);
  });
});

const revealElements = document.querySelectorAll(".reveal");
if ("IntersectionObserver" in window && !reducedMotion) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 },
  );
  revealElements.forEach((element) => observer.observe(element));
} else {
  revealElements.forEach((element) => element.classList.add("is-visible"));
}

const demoStages = Array.from(document.querySelectorAll("[data-demo-stage]"));
const heroPlan = document.querySelector("[data-hero-plan]");
const heroMessages = [
  "Profiling serving-mix",
  "Building eligible plans",
  "Measuring serial latency",
  "Rejecting parity failure",
  "Selecting FX + Inductor",
  "Caching guarded decision",
];
let demoStageIndex = reducedMotion ? heroMessages.length - 1 : 0;

const renderDemoStage = () => {
  demoStages.forEach((stage, index) => {
    stage.classList.toggle("is-active", index === demoStageIndex);
  });
  if (heroPlan) heroPlan.textContent = heroMessages[demoStageIndex];
};

renderDemoStage();
if (!reducedMotion && demoStages.length) {
  window.setInterval(() => {
    demoStageIndex = (demoStageIndex + 1) % heroMessages.length;
    renderDemoStage();
  }, 1000);
}

const useTabs = Array.from(document.querySelectorAll("[data-use-tab]"));
const usePanels = Array.from(document.querySelectorAll("[data-use-panel]"));

const activateUseCase = (name, { focus = false } = {}) => {
  useTabs.forEach((tab) => {
    const selected = tab.dataset.useTab === name;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    if (selected && focus) tab.focus();
  });
  usePanels.forEach((panel) => {
    const selected = panel.dataset.usePanel === name;
    panel.hidden = !selected;
    panel.classList.toggle("is-active", selected);
  });
};

useTabs.forEach((tab, index) => {
  tab.tabIndex = tab.getAttribute("aria-selected") === "true" ? 0 : -1;
  tab.addEventListener("click", () => activateUseCase(tab.dataset.useTab));
  tab.addEventListener("keydown", (event) => {
    let targetIndex = index;
    if (event.key === "ArrowRight") targetIndex = (index + 1) % useTabs.length;
    else if (event.key === "ArrowLeft") targetIndex = (index - 1 + useTabs.length) % useTabs.length;
    else if (event.key === "Home") targetIndex = 0;
    else if (event.key === "End") targetIndex = useTabs.length - 1;
    else return;
    event.preventDefault();
    activateUseCase(useTabs[targetIndex].dataset.useTab, { focus: true });
  });
});

document.querySelectorAll("[data-year]").forEach((element) => {
  element.textContent = String(new Date().getFullYear());
});

window.addEventListener("DOMContentLoaded", () => window.lucide?.createIcons());
