import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import './styles.css'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Admin from './pages/Admin'
import About from './pages/About'

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-black to-gray-900 text-white">
        <nav className="flex items-center justify-between px-8 py-4 bg-white/5 backdrop-blur sticky top-0">
          <Link to="/" className="text-2xl font-extrabold tracking-tight">
            <span className="text-accent">Quantum</span><span className="text-primary">AI</span>
          </Link>
          <div className="space-x-6 text-sm">
            <Link to="/about" className="hover:text-accent">About</Link>
            <Link to="/dashboard" className="hover:text-accent">Dashboard</Link>
            <Link to="/admin" className="hover:text-accent">Admin</Link>
            <Link to="/login" className="px-3 py-1 rounded bg-primary/80 hover:bg-primary">Login</Link>
          </div>
        </nav>
        <div className="p-8">
          <Routes>
            <Route path="/" element={<Landing/>} />
            <Route path="/login" element={<Login/>} />
            <Route path="/dashboard" element={<Dashboard/>} />
            <Route path="/admin" element={<Admin/>} />
            <Route path="/about" element={<About/>} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}

function Landing() {
  return (
    <div className="grid lg:grid-cols-2 gap-10 items-center">
      <div>
        <h1 className="text-5xl font-extrabold leading-tight">
          Precision oncology powered by
          <span className="block text-transparent bg-clip-text bg-gradient-to-r from-accent to-primary">EffResNet-ViT</span>
        </h1>
        <p className="mt-6 text-gray-300 max-w-xl">
          Upload scans. Get AI-assisted brain and lung cancer predictions in seconds. Built for clinicians, trusted by administrators, and inspiring for investors.
        </p>
        <div className="mt-8 space-x-4">
          <Link to="/dashboard" className="px-5 py-3 rounded bg-accent text-black font-semibold hover:opacity-90">Try Demo</Link>
          <Link to="/about" className="px-5 py-3 rounded border border-white/20 hover:bg-white/5">Learn more</Link>
        </div>
      </div>
      <div className="relative">
        <div className="absolute -inset-4 bg-primary/30 blur-3xl rounded-full animate-pulse"/>
        <div className="relative bg-white/5 rounded-xl p-8 shadow-xl border border-white/10">
          <h3 className="text-xl font-semibold mb-3">Live metrics</h3>
          <div className="grid grid-cols-3 gap-4">
            <Metric value="98.4%" label="AUC"/>
            <Metric value=">95%" label="Precision"/>
            <Metric value="90ms" label="Latency"/>
          </div>
        </div>
      </div>
    </div>
  )
}

function Metric({value, label}){
  return (
    <div className="bg-black/30 rounded-lg p-4 border border-white/10">
      <div className="text-2xl font-bold text-accent">{value}</div>
      <div className="text-xs text-gray-400">{label}</div>
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App />)
