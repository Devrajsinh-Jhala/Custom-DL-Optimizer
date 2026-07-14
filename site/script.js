const header = document.querySelector("[data-header]");
const nav = document.querySelector("[data-nav]");
const navToggle = document.querySelector("[data-nav-toggle]");

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

const writeClipboard = async (text) => {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.append(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) throw new Error("Clipboard copy was rejected");
};

const copyText = async (text, button) => {
  const defaultLabel = button.getAttribute("aria-label") ?? "Copy";
  try {
    await writeClipboard(text);
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
if ("IntersectionObserver" in window) {
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

const race = document.querySelector("[data-candidate-race]");
const raceStatus = document.querySelector("[data-decision-status]");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let raceTimers = [];

const clearRaceTimers = () => {
  raceTimers.forEach((timer) => window.clearTimeout(timer));
  raceTimers = [];
};

const setRacePhase = (phase, status) => {
  if (!race) return;
  race.classList.toggle("is-measuring", phase !== "reset");
  race.classList.toggle("is-decided", phase === "decided");
  race.querySelector('[data-candidate="external"]')?.classList.toggle(
    "is-rejected",
    phase === "validating" || phase === "decided",
  );
  race.querySelector('[data-candidate="compiler"]')?.classList.toggle(
    "is-selected",
    phase === "decided",
  );
  if (raceStatus) raceStatus.textContent = status;
};

const runRace = () => {
  clearRaceTimers();
  setRacePhase("reset", "Preparing candidates");
  raceTimers.push(window.setTimeout(() => setRacePhase("measuring", "Measuring steady-state latency"), 350));
  raceTimers.push(window.setTimeout(() => setRacePhase("validating", "External provider rejected: parity failed"), 2100));
  raceTimers.push(window.setTimeout(() => setRacePhase("decided", "FX + Inductor selected with native fallback ready"), 3500));
  raceTimers.push(window.setTimeout(runRace, 7200));
};

if (race) {
  if (reducedMotion) {
    setRacePhase("decided", "FX + Inductor selected with native fallback ready");
  } else {
    runRace();
  }
}

document.querySelectorAll("[data-year]").forEach((element) => {
  element.textContent = String(new Date().getFullYear());
});

window.addEventListener("DOMContentLoaded", () => window.lucide?.createIcons());
