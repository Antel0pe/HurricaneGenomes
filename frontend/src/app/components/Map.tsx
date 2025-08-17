"use client";

import dynamic from "next/dynamic";
import type { ComponentType } from "react";

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

type Props = { tracks?: number[][][] };

export default function Map({ tracks = [] }: Props) {
  return (
    <div className="w-full h-full">
      <MapContainer center={[25.7617, -80.1918]} zoom={5} scrollWheelZoom style={{ height: "100%", width: "100%" }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        {tracks.map((coords, idx) => (
          <Polyline key={idx} positions={coords as unknown as [number, number][]} />
        ))}
      </MapContainer>
    </div>
  );
}


