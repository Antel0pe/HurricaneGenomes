"use client";

import { useEffect, useMemo, useState } from "react";
import ListSearch from "./ListSearch";
import TableWithAction from "./TableWithAction";
import type { DisplayTrack, Track } from "@/types";
import { randomColor } from "@/lib/color";

type Props = {
  setTracks: (tracks: DisplayTrack[]) => void;
  selectedKeys: string[];
  setSelectedKeys: (keys: string[]) => void;
};

export default function Sidebar({ setTracks, selectedKeys, setSelectedKeys }: Props) {
  const [trackMap, setTrackMap] = useState<Record<string, Track>>({});
  const [colorMap, setColorMap] = useState<Record<string, string>>({});

  useEffect(() => {
    const api = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(api + "/tracks", { cache: "no-store" })
      .then((r) => r.json())
      .then((data) => setTrackMap(data || {}))
      .catch(() => setTrackMap({}));
  }, []);

  const keys = useMemo(() => Object.keys(trackMap), [trackMap]);

  const handleClick = (key: string) => {
    const nextKeys = Array.from(new Set([...selectedKeys, key]));
    setSelectedKeys(nextKeys);

    const hasColor = colorMap[key];
    const nextColorMap = hasColor ? colorMap : { ...colorMap, [key]: randomColor() };
    setColorMap(nextColorMap);

    const nextTracks: DisplayTrack[] = nextKeys
      .map((k) => {
        const t = trackMap[k];
        if (!t) return null;
        return { track: t, color: nextColorMap[k] } as DisplayTrack;
      })
      .filter(Boolean) as DisplayTrack[];
    setTracks(nextTracks);
  };

  const handleRemove = (key: string) => {
    const nextKeys = selectedKeys.filter((k) => k !== key);
    setSelectedKeys(nextKeys);
    const nextTracks: DisplayTrack[] = nextKeys
      .map((k) => {
        const t = trackMap[k];
        const c = colorMap[k];
        if (!t || !c) return null;
        return { track: t, color: c } as DisplayTrack;
      })
      .filter(Boolean) as DisplayTrack[];
    setTracks(nextTracks);
  };

  return (
    <aside className="h-full w-full p-2 border-r border-gray-800">
      <h2 className="text-lg font-semibold mb-2">Tracks</h2>
      <ListSearch items={keys} onClick={handleClick} placeholder="Search tracks" />
      <div className="mt-3" />
      <TableWithAction items={selectedKeys} actionLabel="Remove" onAction={handleRemove} colors={colorMap} />
    </aside>
  );
}


