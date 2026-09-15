const form = document.getElementById("cost-form");
const submitBtn = document.getElementById("submit-btn");
const amountEl = document.getElementById("amount");
const linesEl = document.getElementById("receipt-lines");
const errorEl = document.getElementById("form-error");

const CATEGORICAL_SELECTS = [
  "Gender", "Chronic_Condition", "Medicine_Name", "Category", "Type",
  "Generic_or_Branded", "Manufacturer", "Severity", "Treatment_Type", "Region",
];

async function loadMetadata() {
  const res = await fetch("/api/metadata");
  if (!res.ok) throw new Error("Could not load form options");
  const meta = await res.json();

  CATEGORICAL_SELECTS.forEach((name) => {
    const select = form.querySelector(`select[name="${name}"]`);
    const options = meta.categorical_features[name] || [];
    select.innerHTML = options
      .map((opt) => `<option value="${opt}">${opt}</option>`)
      .join("");
  });
}

function setLine(label, value) {
  const row = document.createElement("div");
  row.className = "line";
  row.innerHTML = `<dt>${label}</dt><dd>${value}</dd>`;
  return row;
}

function formatINR(value) {
  return "₹ " + value.toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorEl.hidden = true;

  const data = Object.fromEntries(new FormData(form).entries());

  submitBtn.disabled = true;
  submitBtn.textContent = "Estimating…";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const result = await res.json();

    if (!res.ok) throw new Error(result.error || "Prediction failed");

    amountEl.textContent = formatINR(result.predicted_payable_cost);
    amountEl.classList.add("is-filled");

    linesEl.innerHTML = "";
    linesEl.appendChild(setLine("Medicine", data.Medicine_Name));
    linesEl.appendChild(setLine("Quantity × duration", `${data.Quantity} × ${data.Duration_Days}d`));
    linesEl.appendChild(setLine("Insurance", `${data.Insurance_Pct}%`));
    linesEl.appendChild(setLine("Discount", `${data.Discount_Pct}%`));
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Estimate cost";
  }
});

loadMetadata().catch((err) => {
  errorEl.textContent = err.message;
  errorEl.hidden = false;
});
