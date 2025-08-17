"use client";

import { useEffect, useMemo, useState } from "react";
import ListSearch from "./ListSearch";
import TableWithAction from "./TableWithAction";

type Props = {
  setTracks: (tracks: number[][][]) => void;
  selectedKeys: string[];
  setSelectedKeys: (keys: string[]) => void;
};

export default function Sidebar({ setTracks, selectedKeys, setSelectedKeys }: Props) {
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
    const nextKeys = Array.from(new Set([...selectedKeys, key]));
    setSelectedKeys(nextKeys);
    const nextTracks = nextKeys.map((k) => trackMap[k]).filter(Boolean) as number[][][];
    setTracks(nextTracks);
  };

  const handleRemove = (key: string) => {
    const nextKeys = selectedKeys.filter((k) => k !== key);
    setSelectedKeys(nextKeys);
    const nextTracks = nextKeys.map((k) => trackMap[k]).filter(Boolean) as number[][][];
    setTracks(nextTracks);
  };

  return (
    <aside className="h-full w-full p-2 border-r border-gray-800">
      <h2 className="text-lg font-semibold mb-2">Tracks</h2>
      <ListSearch items={keys} onClick={handleClick} placeholder="Search tracks" />
      <div className="mt-3" />
      <TableWithAction items={selectedKeys} actionLabel="Remove" onAction={handleRemove} />
    </aside>
  );
}


