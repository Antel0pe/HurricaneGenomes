"use client";

import { useState } from "react";
import Map from "./components/Map";
import Sidebar from "./components/Sidebar";

export default function Home() {
  const [tracks, setTracks] = useState<number[][][]>([]);
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  return (
    <main className="flex h-screen w-screen">
      <div className="w-1/4 h-full">
        <Sidebar setTracks={setTracks} selectedKeys={selectedKeys} setSelectedKeys={setSelectedKeys} />
      </div>
      <div className="flex-1 h-full">
        <Map tracks={tracks} />
      </div>
    </main>
  );
}
