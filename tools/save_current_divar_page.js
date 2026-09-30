(() => {
  const now = new Date().toISOString().replace(/[:.]/g, "-");
  const district = (location.pathname.split("/").filter(Boolean).pop() || "divar").replace(/[^a-zA-Z0-9_-]+/g, "_");
  const filename = `divar_${district}_${now}.html`;

  // No network request is made. This saves the already-loaded public page exactly
  // as the browser currently has it, including script tags such as
  // window.__PRELOADED_STATE__ and JSON-LD when present.
  const html = "<!doctype html>\n" + document.documentElement.outerHTML;
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);

  console.log(
    `Saved ${filename}. Upload it to the Divar Scanner Colab with ACQUISITION_MODE='snapshot'.`
  );
})();