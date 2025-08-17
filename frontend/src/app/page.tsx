import Map from "./components/Map";
import Sidebar from "./components/Sidebar";

export default function Home() {
  return (
    <main className="flex h-screen w-screen">
      <div className="w-1/4 h-full">
        <Sidebar />
      </div>
      <div className="flex-1 h-full">
        <Map />
      </div>
    </main>
  );
}
