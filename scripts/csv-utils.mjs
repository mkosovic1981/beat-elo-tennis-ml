// Minimal RFC4180-ish CSV parser (handles quoted fields with embedded commas/newlines/escaped quotes).
// Used because matches_export.csv has free-text Trenutci fields with commas inside quotes.
export function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;
  let i = 0;
  const len = text.length;
  while (i < len) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        } else {
          inQuotes = false;
          i++;
          continue;
        }
      } else {
        field += c;
        i++;
        continue;
      }
    } else {
      if (c === '"') {
        inQuotes = true;
        i++;
        continue;
      } else if (c === ',') {
        row.push(field);
        field = '';
        i++;
        continue;
      } else if (c === '\r') {
        i++;
        continue;
      } else if (c === '\n') {
        row.push(field);
        rows.push(row);
        row = [];
        field = '';
        i++;
        continue;
      } else {
        field += c;
        i++;
        continue;
      }
    }
  }
  // last field/row (in case file doesn't end with newline)
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}
