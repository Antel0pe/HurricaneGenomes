"use client";

import { useEffect, useMemo, useState } from "react";
import ListSearch from "./ListSearch";

type Props = {
  setTracks: (tracks: number[][][]) => void;
};

export default function Sidebar({ setTracks }: Props) {
  const [trackMap, setTrackMap] = useState<Record<string, number[][]>>({});

  useEffect(() => {
    const api = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(api + "/tracks", { cache: "no-store" })
      .then((r) => r.json())
      .then((data) => setTrackMap(data || {}))
      .catch(() => setTrackMap({}));
  }, []);

  const keys = useMemo(() => Object.keys(trackMap), [trackMap]);

  const handleClick = (key: string) => {
    const coords = trackMap[key] || [];
    if (coords.length) setTracks([coords]);
  };

  return (
    <aside className="h-full w-full p-2 border-r border-gray-800">
      <h2 className="text-lg font-semibold mb-2">Tracks</h2>
      <ListSearch items={keys} onClick={handleClick} placeholder="Search tracks" />
    </aside>
  );
}


