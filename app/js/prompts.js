/* FindMine – every sentence the app speaks.
 *
 * Languages: en (English), fr (French), ar (Standard Arabic), dz (Algerian Darija, Arabic script)
 * Verbosity: "short" (few words, for guidance while moving) or "detailed" (full sentences)
 *
 * Use in a page:
 *   <script src="prompts.js"></script>
 *   FM.setLang("dz");              // saved on the phone
 *   FM.setVerbosity("short");      // saved on the phone
 *   FM.say("at_clock", { item: "keys", clock: 2 });
 *   const text = FM.t("searching", { item: "keys" });
 *
 * A phrase is either one string (same for both verbosities) or { short, detailed }.
 * {item}, {clock}, {n}, {total}, {side}, {dist}, {items} are filled in automatically.
 * Missing Darija phrases fall back to Arabic, then English.
 */
(function () {
  const PROMPTS = {
    // ───────────────────────── ENGLISH ─────────────────────────
    en: {
      // App
      welcome:          { short: "FindMine ready.", detailed: "FindMine is ready. Say: find my keys." },
      disclaimer:       "FindMine helps you find your objects. It is not a navigation or mobility aid.",
      camera_on:        { short: "Camera on.", detailed: "Camera is on. Move the phone slowly around the room." },
      camera_blocked:   "I can’t use the camera. Please allow camera access.",
      server_lost:      { short: "Connection lost.", detailed: "Connection lost. Trying again." },
      server_back:      "Connected again.",
      listening:        "Listening.",
      didnt_understand: { short: "Sorry, again?", detailed: "Sorry, I didn’t understand. Please say it again." },

      // Camera coach
      too_dark:         { short: "Too dark.", detailed: "It’s too dark. Please turn on a light." },
      too_bright:       { short: "Too bright.", detailed: "Too much light. Turn away from the window or lamp." },
      blurry:           { short: "Blurry.", detailed: "The picture is blurry. Hold the phone still for a moment." },
      shaking:          { short: "Slower.", detailed: "You’re moving too fast. Move the phone more slowly." },
      lens_blocked:     { short: "Camera covered.", detailed: "The camera is covered. Move your finger away from the lens." },
      good_view:        "Good, keep going.",

      // Registration
      side_front:       "the front",
      side_back:        "the back",
      side_left:        "the left side",
      side_right:       "the right side",
      side_top:         "from above, close up",
      shot:             { short: "Photo {n}: {side}.", detailed: "Photo {n} of {total}: {side}. Hold the item close so it fills the screen." },
      retake:           { short: "Retake, closer.", detailed: "Not enough detail. Move closer or try another angle." },
      photo_saved:      "Photo saved.",
      item_saved:       "{item} is saved with {total} photos.",

      // Search
      searching:        { short: "Searching for {item}.", detailed: "Searching for {item}. Move the phone slowly around the room." },
      scan_slowly:      { short: "Keep scanning.", detailed: "Keep moving the phone slowly, left and right." },
      not_found_yet:    { short: "Not yet.", detailed: "I don’t see it yet. Try lower, near the floor, or on tables." },
      give_up:          "I can’t find {item} here. Try another room.",
      item_unknown:     "I don’t know that item. Your items are: {items}.",
      no_items:         "You have no registered items yet.",

      // Guidance
      at_clock:         { short: "{clock} o’clock.", detailed: "{item} at {clock} o’clock, {dist}." },
      left:             { short: "Left.", detailed: "Turn a little to the left." },
      right:            { short: "Right.", detailed: "Turn a little to the right." },
      ahead:            { short: "Straight ahead.", detailed: "It’s straight ahead of you." },
      up:               { short: "Higher.", detailed: "Tilt the phone up a little." },
      down:             { short: "Lower.", detailed: "Tilt the phone down a little." },
      closer:           { short: "Closer.", detailed: "Walk slowly closer." },
      stop:             { short: "Stop.", detailed: "Stop. It’s right in front of you." },
      lost_track:       { short: "Lost it.", detailed: "I lost it. Move back slowly to where you were." },
      dist_far:         "more than two meters away",
      dist_mid:         "about one meter away",
      dist_near:        "within arm’s reach",

      // Verification
      is_this_yours:    { short: "Is this {item}?", detailed: "I see something that looks like {item}. Is it yours? Say yes or no." },
      found:            { short: "Found {item}!", detailed: "Found {item}! It’s right in front of you." },
      not_sure:         { short: "Not sure.", detailed: "I’m not sure it’s yours. Let me look again." },
      say_yes_no:       "Please say yes or no.",
    },

    // ───────────────────────── FRANÇAIS ─────────────────────────
    fr: {
      welcome:          { short: "FindMine prêt.", detailed: "FindMine est prêt. Dites : trouve mes clés." },
      disclaimer:       "FindMine vous aide à retrouver vos objets. Ce n’est pas une aide à la navigation ni à la mobilité.",
      camera_on:        { short: "Caméra activée.", detailed: "La caméra est activée. Bougez le téléphone lentement autour de la pièce." },
      camera_blocked:   "Je ne peux pas utiliser la caméra. Autorisez l’accès à la caméra.",
      server_lost:      { short: "Connexion perdue.", detailed: "Connexion perdue. Je réessaie." },
      server_back:      "Connexion rétablie.",
      listening:        "Je vous écoute.",
      didnt_understand: { short: "Pardon ?", detailed: "Désolé, je n’ai pas compris. Pouvez-vous répéter ?" },

      too_dark:         { short: "Trop sombre.", detailed: "Il fait trop sombre. Allumez une lumière." },
      too_bright:       { short: "Trop de lumière.", detailed: "Trop de lumière. Détournez-vous de la fenêtre ou de la lampe." },
      blurry:           { short: "Flou.", detailed: "L’image est floue. Tenez le téléphone immobile un instant." },
      shaking:          { short: "Plus lentement.", detailed: "Vous bougez trop vite. Bougez le téléphone plus lentement." },
      lens_blocked:     { short: "Caméra cachée.", detailed: "La caméra est cachée. Retirez votre doigt de l’objectif." },
      good_view:        "C’est bien, continuez.",

      side_front:       "l’avant",
      side_back:        "l’arrière",
      side_left:        "le côté gauche",
      side_right:       "le côté droit",
      side_top:         "le dessus, de près",
      shot:             { short: "Photo {n} : {side}.", detailed: "Photo {n} sur {total} : {side}. Tenez l’objet près pour qu’il remplisse l’écran." },
      retake:           { short: "Reprenez, plus près.", detailed: "Pas assez de détails. Rapprochez-vous ou changez d’angle." },
      photo_saved:      "Photo enregistrée.",
      item_saved:       "{item} est enregistré avec {total} photos.",

      searching:        { short: "Je cherche {item}.", detailed: "Je cherche {item}. Bougez le téléphone lentement autour de la pièce." },
      scan_slowly:      { short: "Continuez.", detailed: "Continuez à bouger le téléphone lentement, à gauche et à droite." },
      not_found_yet:    { short: "Pas encore.", detailed: "Je ne le vois pas encore. Essayez plus bas, près du sol, ou sur les tables." },
      give_up:          "Je ne trouve pas {item} ici. Essayez une autre pièce.",
      item_unknown:     "Je ne connais pas cet objet. Vos objets sont : {items}.",
      no_items:         "Vous n’avez encore aucun objet enregistré.",

      at_clock:         { short: "À {clock} heures.", detailed: "{item} à {clock} heures, {dist}." },
      left:             { short: "À gauche.", detailed: "Tournez un peu à gauche." },
      right:            { short: "À droite.", detailed: "Tournez un peu à droite." },
      ahead:            { short: "Tout droit.", detailed: "C’est juste devant vous." },
      up:               { short: "Plus haut.", detailed: "Levez un peu le téléphone." },
      down:             { short: "Plus bas.", detailed: "Baissez un peu le téléphone." },
      closer:           { short: "Approchez.", detailed: "Avancez doucement." },
      stop:             { short: "Stop.", detailed: "Stop. C’est juste devant vous." },
      lost_track:       { short: "Perdu.", detailed: "Je l’ai perdu. Revenez doucement à votre position précédente." },
      dist_far:         "à plus de deux mètres",
      dist_mid:         "à environ un mètre",
      dist_near:        "à portée de main",

      is_this_yours:    { short: "C’est {item} ?", detailed: "Je vois quelque chose qui ressemble à {item}. C’est à vous ? Dites oui ou non." },
      found:            { short: "{item} trouvé !", detailed: "{item} trouvé ! C’est juste devant vous." },
      not_sure:         { short: "Pas sûr.", detailed: "Je ne suis pas sûr que ce soit à vous. Je regarde encore." },
      say_yes_no:       "Dites oui ou non, s’il vous plaît.",
    },

    // ───────────────────── العربية الفصحى ─────────────────────
    ar: {
      welcome:          { short: "التطبيق جاهز.", detailed: "التطبيق جاهز. قل: ابحث عن مفاتيحي." },
      disclaimer:       "يساعدك FindMine على العثور على أغراضك. وهو ليس أداة للتنقل أو للمساعدة على الحركة.",
      camera_on:        { short: "الكاميرا تعمل.", detailed: "الكاميرا تعمل. حرّك الهاتف ببطء في أرجاء الغرفة." },
      camera_blocked:   "لا أستطيع استخدام الكاميرا. اسمح بالوصول إلى الكاميرا.",
      server_lost:      { short: "انقطع الاتصال.", detailed: "انقطع الاتصال. أحاول مرة أخرى." },
      server_back:      "عاد الاتصال.",
      listening:        "أنا أستمع.",
      didnt_understand: { short: "عفوًا، أعد؟", detailed: "عفوًا، لم أفهم. من فضلك أعد الكلام." },

      too_dark:         { short: "مظلم جدًا.", detailed: "المكان مظلم جدًا. من فضلك أشعل الضوء." },
      too_bright:       { short: "ضوء قوي.", detailed: "الضوء قوي جدًا. ابتعد عن النافذة أو المصباح." },
      blurry:           { short: "الصورة غير واضحة.", detailed: "الصورة غير واضحة. ثبّت الهاتف لحظة." },
      shaking:          { short: "ببطء.", detailed: "أنت تتحرك بسرعة. حرّك الهاتف ببطء." },
      lens_blocked:     { short: "الكاميرا مغطاة.", detailed: "الكاميرا مغطاة. أبعد إصبعك عن العدسة." },
      good_view:        "جيد، استمر.",

      side_front:       "الواجهة الأمامية",
      side_back:        "الجهة الخلفية",
      side_left:        "الجانب الأيسر",
      side_right:       "الجانب الأيمن",
      side_top:         "من الأعلى، عن قرب",
      shot:             { short: "الصورة {n}: {side}.", detailed: "الصورة {n} من {total}: {side}. قرّب الشيء حتى يملأ الشاشة." },
      retake:           { short: "أعد، أقرب.", detailed: "التفاصيل غير كافية. اقترب أو غيّر الزاوية." },
      photo_saved:      "حُفظت الصورة.",
      item_saved:       "تم حفظ {item} بـ {total} صور.",

      searching:        { short: "أبحث عن {item}.", detailed: "أبحث عن {item}. حرّك الهاتف ببطء في أرجاء الغرفة." },
      scan_slowly:      { short: "استمر.", detailed: "استمر في تحريك الهاتف ببطء، يمينًا ويسارًا." },
      not_found_yet:    { short: "ليس بعد.", detailed: "لا أراه بعد. جرّب في الأسفل، قرب الأرض، أو على الطاولات." },
      give_up:          "لا أجد {item} هنا. جرّب غرفة أخرى.",
      item_unknown:     "لا أعرف هذا الشيء. أشياؤك هي: {items}.",
      no_items:         "ليس لديك أشياء مسجّلة بعد.",

      at_clock:         { short: "عند الساعة {clock}.", detailed: "{item} عند الساعة {clock}، {dist}." },
      left:             { short: "يسارًا.", detailed: "استدر قليلًا إلى اليسار." },
      right:            { short: "يمينًا.", detailed: "استدر قليلًا إلى اليمين." },
      ahead:            { short: "أمامك.", detailed: "إنه أمامك مباشرة." },
      up:               { short: "أعلى.", detailed: "ارفع الهاتف قليلًا." },
      down:             { short: "أسفل.", detailed: "اخفض الهاتف قليلًا." },
      closer:           { short: "اقترب.", detailed: "تقدّم ببطء." },
      stop:             { short: "توقف.", detailed: "توقف. إنه أمامك مباشرة." },
      lost_track:       { short: "فقدته.", detailed: "فقدته. ارجع ببطء إلى مكانك السابق." },
      dist_far:         "على بعد أكثر من مترين",
      dist_mid:         "على بعد متر تقريبًا",
      dist_near:        "في متناول يدك",

      is_this_yours:    { short: "هل هذا {item}؟", detailed: "أرى شيئًا يشبه {item}. هل هو لك؟ قل نعم أو لا." },
      found:            { short: "وجدت {item}!", detailed: "وجدت {item}! إنه أمامك مباشرة." },
      not_sure:         { short: "لست متأكدًا.", detailed: "لست متأكدًا أنه لك. سأنظر مرة أخرى." },
      say_yes_no:       "من فضلك قل نعم أو لا.",
    },

    // ─────────────────── الدارجة الجزائرية ───────────────────
    // Please review: natural Darija differs by region.
    dz: {
      welcome:          { short: "FindMine واجد.", detailed: "FindMine واجد. قول: وين السوارت تاعي." },
      disclaimer:       "FindMine يعاونك تلقى الحوايج تاعك. ماشي باش يوجهك في الطريق ولا يعاونك في المشي.",
      camera_on:        { short: "الكاميرا راهي خدامة.", detailed: "الكاميرا راهي خدامة. دوّر التيليفون بالشوية في البيت." },
      camera_blocked:   "ما نقدرش نستعمل الكاميرا. عطيني البيرميسيون تاع الكاميرا.",
      server_lost:      { short: "الكونيكسيون طاحت.", detailed: "الكونيكسيون طاحت. راني نعاود." },
      server_back:      "الكونيكسيون رجعت.",
      listening:        "راني نسمع فيك.",
      didnt_understand: { short: "سمحلي، عاود؟", detailed: "سمحلي، ما فهمتكش. عاود قولها." },

      too_dark:         { short: "ظلام بزاف.", detailed: "ظلام بزاف. شعل الضو من فضلك." },
      too_bright:       { short: "ضو بزاف.", detailed: "كاين ضو بزاف. بعّد على الطاقة ولا اللامبة." },
      blurry:           { short: "الصورة مغبشة.", detailed: "الصورة مغبشة. حبس التيليفون شوية ما تحركوش." },
      shaking:          { short: "بالشوية.", detailed: "راك تتحرك بالزربة. حرك التيليفون بالشوية." },
      lens_blocked:     { short: "الكاميرا مغطية.", detailed: "الكاميرا مغطية. نحّي صبعك من الكاميرا." },
      good_view:        "مليح، كمّل.",

      side_front:       "القدام",
      side_back:        "اللور",
      side_left:        "الجنب ليسر",
      side_right:       "الجنب ليمن",
      side_top:         "من الفوق، قريب",
      shot:             { short: "التصويرة {n}: {side}.", detailed: "التصويرة {n} من {total}: {side}. قرّب الحاجة باش تعمّر الشاشة." },
      retake:           { short: "عاود، قرّب.", detailed: "ما بانتش مليح. قرّب ولا بدّل الجيهة." },
      photo_saved:      "التصويرة تسجلت.",
      item_saved:       "{item} تسجلت بـ {total} تصاور.",

      searching:        { short: "راني نقلب على {item}.", detailed: "راني نقلب على {item}. دوّر التيليفون بالشوية في البيت." },
      scan_slowly:      { short: "كمّل.", detailed: "كمّل دوّر التيليفون بالشوية، ليمين وليسار." },
      not_found_yet:    { short: "مازال.", detailed: "مازال ما شفتهاش. جرب لتحت، قريب من الأرض، ولا فوق الطبلة." },
      give_up:          "ما لقيتش {item} هنا. جرب في بيت أخرى.",
      item_unknown:     "ما نعرفش هاد الحاجة. الحوايج تاعك هوما: {items}.",
      no_items:         "مازال ما سجلت حتى حاجة.",

      at_clock:         { short: "على {clock}.", detailed: "{item} على الساعة {clock}، {dist}." },
      left:             { short: "ليسار.", detailed: "دور شوية ليسار." },
      right:            { short: "ليمين.", detailed: "دور شوية ليمين." },
      ahead:            { short: "قدامك.", detailed: "راهي قدامك نيشان." },
      up:               { short: "الفوق.", detailed: "هز التيليفون شوية للفوق." },
      down:             { short: "لتحت.", detailed: "هبط التيليفون شوية لتحت." },
      closer:           { short: "قرّب.", detailed: "قرّب بالشوية." },
      stop:             { short: "حبس.", detailed: "حبس. راهي قدامك نيشان." },
      lost_track:       { short: "ضيعتها.", detailed: "ضيعتها. ولّي بالشوية للبلاصة اللي كنت فيها." },
      dist_far:         "بعيدة كثر من زوج ميترو",
      dist_mid:         "بعيدة قريب ميترو",
      dist_near:        "قريبة ليك، تقدر تحكمها بيدك",

      is_this_yours:    { short: "هادي {item}؟", detailed: "راني نشوف حاجة تشبه لـ {item}. تاعك؟ قول إيه ولا لا." },
      found:            { short: "لقيت {item}!", detailed: "لقيت {item}! راهي قدامك نيشان." },
      not_sure:         { short: "مارانيش متأكد.", detailed: "مارانيش متأكد بلي تاعك. خليني نشوف مرة أخرى." },
      say_yes_no:       "قول إيه ولا لا من فضلك.",
    },
  };

  const LANG_INFO = {
    en: { name: "English",  tts: "en-US", listen: "en-US", dir: "ltr" },
    fr: { name: "Français", tts: "fr-FR", listen: "fr-FR", dir: "ltr" },
    ar: { name: "العربية",  tts: "ar-SA", listen: "ar-SA", dir: "rtl" },
    dz: { name: "الدارجة",  tts: "ar-SA", listen: "ar-DZ", dir: "rtl" },  // no Darija voice exists: Arabic voice reads it
  };
  const FALLBACK = { dz: ["dz", "ar", "en"], ar: ["ar", "en"], fr: ["fr", "en"], en: ["en"] };

  function load(key, def) { try { return localStorage.getItem(key) || def; } catch (e) { return def; } }
  function store(key, val) { try { localStorage.setItem(key, val); } catch (e) {} }

  let lang = load("findmine-lang", "en");
  let verbosity = load("findmine-verbosity", "detailed");
  let mode = load("findmine-mode", "full");        // "full" or "baseline" (test condition A)
  if (!PROMPTS[lang]) lang = "en";

  function raw(key, l) {
    for (const code of FALLBACK[l] || ["en"]) {
      const p = PROMPTS[code] && PROMPTS[code][key];
      if (p !== undefined) return p;
    }
    return key;
  }

  function fill(text, params, l) {
    return text.replace(/\{(\w+)\}/g, (_, name) => {
      if (!(name in params)) return "";
      // {side} and {dist} can be phrase keys, e.g. side: "side_front"
      const v = params[name];
      return typeof v === "string" && PROMPTS.en[v] !== undefined ? t(v, {}, { lang: l, verbosity: "short" }) : String(v);
    });
  }

  function t(key, params = {}, opts = {}) {
    const l = opts.lang || lang;
    const vb = opts.verbosity || verbosity;
    const p = raw(key, l);
    const text = typeof p === "string" ? p : (p[vb] || p.detailed || p.short);
    return fill(text, params, l);
  }

  function pickVoice(code) {
    if (!("speechSynthesis" in window)) return null;
    const voices = speechSynthesis.getVoices();
    const base = code.slice(0, 2);
    return voices.find((v) => v.lang.replace("_", "-") === code) ||
           voices.find((v) => v.lang.toLowerCase().startsWith(base)) || null;
  }
  if ("speechSynthesis" in window) speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();

  // interrupt: true cuts off whatever is being said (use for urgent guidance like "stop")
  function say(key, params = {}, opts = {}) {
    const text = t(key, params, opts);
    if (!("speechSynthesis" in window)) return text;
    if (opts.interrupt) speechSynthesis.cancel();
    const info = LANG_INFO[opts.lang || lang];
    const u = new SpeechSynthesisUtterance(text);
    u.lang = info.tts;
    const v = pickVoice(info.tts);
    if (v) u.voice = v;
    if (opts.rate) u.rate = opts.rate;
    speechSynthesis.speak(u);
    return text;
  }

  function hasVoice(l = lang) { return !!pickVoice(LANG_INFO[l].tts); }

  window.FM = {
    PROMPTS, LANG_INFO,
    t, say, hasVoice,
    get lang() { return lang; },
    get verbosity() { return verbosity; },
    setLang(l) { if (PROMPTS[l]) { lang = l; store("findmine-lang", l); } },
    setVerbosity(v) { if (v === "short" || v === "detailed") { verbosity = v; store("findmine-verbosity", v); } },
    get mode() { return mode; },
    setMode(m) { if (m === "full" || m === "baseline") { mode = m; store("findmine-mode", m); } },
    // Extra fields for the /ws "start" message, e.g. { mode: "baseline" }
    startExtras() { return mode === "baseline" ? { mode: "baseline" } : {}; },
    showDisclaimer,
    pickVoice,
    disclaimerSeen() { return load("findmine-disclaimer-seen", "") === "1"; },
    markDisclaimerSeen() { store("findmine-disclaimer-seen", "1"); },
  };

  // ───── First-open disclaimer ─────
  // Browsers only allow speech after a tap, so the first screen is one big "tap anywhere" button.
  function showDisclaimer(force) {
    if (!force && load("findmine-disclaimer-seen", "") === "1") return;
    if (document.getElementById("fm-disclaimer")) return;
    const info = LANG_INFO[lang];
    const box = document.createElement("button");
    box.id = "fm-disclaimer";
    box.type = "button";
    box.dir = info.dir;
    box.setAttribute("aria-label", t("disclaimer") + " " + tapLabel());
    box.style.cssText = "position:fixed;inset:0;z-index:9999;border:0;margin:0;width:100%;height:100%;" +
      "display:flex;flex-direction:column;align-items:center;justify-content:center;gap:24px;" +
      "padding:calc(env(safe-area-inset-top,0px) + 24px) 28px calc(env(safe-area-inset-bottom,0px) + 24px);" +
      "background:var(--bg,#FFF8EA);color:var(--text,#1E2252);font:700 1.6rem/1.4 \"Atkinson Hyperlegible\",system-ui,-apple-system,Segoe UI,Roboto,sans-serif;" +
      "text-align:center;cursor:pointer;";
    box.innerHTML = '<span style="font-size:3rem" aria-hidden="true">🔎</span>' +
      '<span></span><span style="font-size:1.1rem;font-weight:400;opacity:.85"></span>';
    box.children[1].textContent = t("disclaimer");
    box.children[2].textContent = tapLabel();
    let spoken = false;
    box.addEventListener("click", () => {
      store("findmine-disclaimer-seen", "1");
      if (!spoken && "speechSynthesis" in window) {
        spoken = true;
        const u = new SpeechSynthesisUtterance(t("disclaimer"));
        u.lang = info.tts;
        const v = pickVoice(info.tts); if (v) u.voice = v;
        u.onend = () => box.remove();
        u.onerror = () => box.remove();
        speechSynthesis.cancel();
        speechSynthesis.speak(u);
        box.children[2].textContent = "";
        setTimeout(() => box.remove(), 9000);       // safety: never stay stuck
      } else {
        speechSynthesis.cancel();
        box.remove();                                // second tap skips
      }
    });
    document.body.appendChild(box);
    box.focus();
  }
  function tapLabel() {
    return { en: "Tap anywhere to continue.", fr: "Touchez l’écran pour continuer.",
             ar: "المس الشاشة للمتابعة.", dz: "اضغط على الشاشة باش تكمل." }[lang];
  }
  // A page can set window.FM_OWN_START = true before loading this file to run its own start screen.
  if (!window.FM_OWN_START) {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => showDisclaimer());
    else showDisclaimer();
  }
})();
