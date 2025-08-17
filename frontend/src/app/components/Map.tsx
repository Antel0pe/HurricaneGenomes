"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";

const MapContainer = dynamic(
  () => import("react-leaflet").then(m => m.MapContainer),
  { ssr: false }
);
const TileLayer = dynamic(
  () => import("react-leaflet").then(m => m.TileLayer),
  { ssr: false }
);
const GeoJSON = dynamic(
  () => import("react-leaflet").then(m => m.GeoJSON),
  { ssr: false }
);

type Props = { tracks?: number[][][] };

export default function Map({ tracks = [] }: Props) {
  const featureCollection = useMemo(() => ({
    type: "FeatureCollection",
    features: tracks.map((coords) => ({
      type: "Feature",
      properties: {},
      geometry: {
        type: "LineString",
        coordinates: coords.map(([lat, lng]) => [lng, lat]),
      },
    })),
  }), [tracks]);

  return (
    <div className="w-full h-full">
      <MapContainer center={[25.7617, -80.1918]} zoom={5} scrollWheelZoom style={{ height: "100%", width: "100%" }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        {tracks.length > 0 && (
          <GeoJSON data={featureCollection as any} />
        )}
      </MapContainer>
    </div>
  );
}


