"use client";

type Props = {
  items: string[];
  actionLabel: string;
  onAction: (value: string) => void;
  colors?: Record<string, string>;
};

export default function TableWithAction({ items, actionLabel, onAction, colors }: Props) {
  return (
    <div className="border border-gray-800 rounded">
      <div className="grid grid-cols-2 text-xs font-semibold px-2 py-1 border-b border-gray-800">
        <div>Track</div>
        <div>Action</div>
      </div>
      <div className="max-h-[35vh] overflow-auto">
        {items.map((i) => (
          <div key={i} className="grid grid-cols-2 items-center px-2 py-1 border-b border-gray-900">
            <div className="truncate pr-2 flex items-center gap-2">
              {colors && colors[i] && (
                <span
                  className="inline-block w-3 h-3 rounded-sm border border-gray-700 shrink-0"
                  style={{ backgroundColor: colors[i] }}
                />
              )}
              <span className="truncate">{i}</span>
            </div>
            <div>
              <button
                onClick={() => onAction(i)}
                className="border border-gray-700 rounded px-2 py-0.5 hover:bg-gray-800 text-xs"
              >
                {actionLabel}
              </button>
            </div>
          </div>
        ))}
        {items.length === 0 && (
          <div className="px-2 py-2 text-xs text-gray-400">No tracks displayed</div>
        )}
      </div>
    </div>
  );
}


