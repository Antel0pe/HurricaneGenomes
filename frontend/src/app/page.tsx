export default async function Home() {
  const api = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  const res = await fetch(api + '/health', { cache: 'no-store' });
  const data = await res.json();
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-6">
      <div className="text-center">
        <h1 className="text-2xl font-semibold">Next.js + FastAPI</h1>
        <p className="mt-2 text-sm text-gray-300">Backend says: <span className="font-mono">{data.message}</span></p>
      </div>
    </main>
  );
}
