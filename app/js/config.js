/* FindMine config: loaded first on every page, in <head>.
 * - Works out where the server is (same address when the server hosts the app, otherwise the default).
 * - Applies the reader's display choices (text size, light/dark) before the page draws.
 */
(function () {
  var DEFAULT_SERVER = "https://54-90-98-230.sslip.io";   // change here after a redeploy to a new IP
  var override = null, text = null, theme = null;
  try {
    override = localStorage.getItem("findmine-server");
    text = localStorage.getItem("findmine-text");
    theme = localStorage.getItem("findmine-theme");
  } catch (e) {}
  var servedByServer = /\.sslip\.io$/.test(location.hostname) || (location.hostname === "localhost" && location.port === "8000");
  var http = (override || (servedByServer ? location.origin : DEFAULT_SERVER)).replace(/\/$/, "");
  window.FM_SERVER = http;

  // This phone's random id: the server lists only the items registered from this phone.
  var owner = null;
  try { owner = localStorage.getItem("findmine-owner"); } catch (e) {}
  if (!owner) {
    owner = "p-" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4);
    try { localStorage.setItem("findmine-owner", owner); } catch (e) {}
  }
  window.FM_OWNER = owner;
  window.FM_WS = http.replace(/^http/, "ws") + "/ws";

  var root = document.documentElement;
  if (text === "large" || text === "xl") root.setAttribute("data-text", text);
  if (theme === "light" || theme === "dark") root.setAttribute("data-theme", theme);

  // Decorative waves on every page (hidden from screen readers)
  var wave = '<svg class="POS" viewBox="0 0 400 300" fill="none" focusable="false">' +
    '<path class="a" d="M10 290 C 90 200, 120 120, 220 120 S 360 40, 390 10" stroke-width="46" stroke-linecap="round"/>' +
    '<path class="b" d="M70 300 C 140 230, 180 170, 260 170 S 380 110, 400 70" stroke-width="30" stroke-linecap="round"/></svg>';
  // iOS Safari only speaks after speech was started by a tap: unlock it on the first touch, silently.
  function unlockSpeech() {
    try { if ("speechSynthesis" in window) { var u = new SpeechSynthesisUtterance(" "); u.volume = 0; speechSynthesis.speak(u); } } catch (e) {}
  }
  document.addEventListener("pointerdown", unlockSpeech, { once: true, capture: true });

  document.addEventListener("DOMContentLoaded", function () {
    document.body.insertAdjacentHTML("afterbegin",
      '<div class="waves" aria-hidden="true">' + wave.replace("POS", "top") + wave.replace("POS", "bottom") + "</div>");
  });
})();
