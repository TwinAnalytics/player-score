const BASE = import.meta.env.BASE_URL;

export function slugify(name) {
  if (!name) return '';
  return name
    .toLowerCase()
    .replace(/\s+/g, '_')
    .replace(/-/g, '_')
    .replace(/[^a-z0-9_]/g, '');
}

export function crestUrl(clubName) {
  if (!clubName) return null;
  return `${BASE}crests/${slugify(clubName)}.png`;
}
