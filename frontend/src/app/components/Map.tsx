"use client";

import dynamic from "next/dynamic";
import type { ComponentType } from "react";
import type { DisplayTrack } from "@/types";

type AnyProps = Record<string, unknown>;

const MapContainer = dynamic(
  () => import("react-leaflet").then(m => m.MapContainer),
  { ssr: false }
) as unknown as ComponentType<AnyProps>;
const TileLayer = dynamic(
  () => import("react-leaflet").then(m => m.TileLayer),
  { ssr: false }
) as unknown as ComponentType<AnyProps>;
const Polyline = dynamic(
  () => import("react-leaflet").then(m => m.Polyline),
  { ssr: false }
) as unknown as ComponentType<AnyProps>;

type Props = { tracks?: DisplayTrack[] };

export default function Map({ tracks = [] }: Props) {
  return (
    <div className="w-full h-full">
      <MapContainer center={[25.7617, -80.1918]} zoom={5} scrollWheelZoom style={{ height: "100%", width: "100%" }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        {tracks.map((t, idx) => (
          <Polyline key={idx} positions={t.track} pathOptions={{ color: t.color }} />
        ))}
      </MapContainer>
    </div>
  );
}


