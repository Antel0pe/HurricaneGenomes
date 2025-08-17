"use client";

import { useMemo, useState } from "react";

type Props = {
  items: string[];
  onClick: (value: string) => void;
  placeholder?: string;
};

export default function ListSearch({ items, onClick, placeholder = "Search..." }: Props) {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.filter((i) => i.toLowerCase().includes(q));
  }, [items, query]);

  return (
    <div className="flex flex-col gap-2">
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={placeholder}
        className="w-full bg-transparent border border-gray-700 rounded px-2 py-1 text-sm focus:outline-none"
      />
      <div className="max-h-[50vh] overflow-auto border border-gray-800 rounded">
        {filtered.map((i) => (
          <button
            key={i}
            onClick={() => onClick(i)}
            className="w-full text-left px-2 py-1 text-sm hover:bg-gray-800"
          >
            {i}
          </button>
        ))}
        {filtered.length === 0 && (
          <div className="px-2 py-2 text-xs text-gray-400">No results</div>
        )}
      </div>
    </div>
  );
}


