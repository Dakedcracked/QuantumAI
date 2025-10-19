import React, { useState } from 'react'
import api from '../lib/api'

export default function Dashboard(){
  const [file, setFile] = useState(null)
  const [model, setModel] = useState('brain')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    if(!file) return
    setLoading(true)
    try{
      const form = new FormData()
      form.append('file', file)
      const res = await api.post(`/api/predictions?model_type=${model}`, form)
      setResult(res.data)
    }catch(err){
      alert('Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="grid md:grid-cols-2 gap-10">
      <div className="bg-white/5 p-6 rounded-xl border border-white/10">
        <h2 className="text-xl font-semibold mb-3">Upload Medical Image</h2>
        <form onSubmit={submit} className="space-y-4">
          <select value={model} onChange={e=>setModel(e.target.value)} className="w-full px-3 py-2 rounded bg-black/40 border border-white/10">
            <option value="brain">Brain Cancer</option>
            <option value="lung">Lung Cancer</option>
          </select>
          <input type="file" onChange={e=>setFile(e.target.files[0])} className="block"/>
          <button disabled={loading} className="px-4 py-2 rounded bg-accent text-black font-semibold disabled:opacity-50">{loading ? 'Predicting...' : 'Predict'}</button>
        </form>
      </div>
      <div className="bg-white/5 p-6 rounded-xl border border-white/10">
        <h2 className="text-xl font-semibold mb-3">Result</h2>
        {result ? (
          <div>
            <div className="text-3xl font-bold text-accent">{result.predicted_label}</div>
            <div className="text-sm text-gray-400">Confidence: {(result.confidence*100).toFixed(2)}%</div>
            <div className="text-xs text-gray-500 break-all mt-2">Image: {result.image_path}</div>
          </div>
        ) : (
          <p className="text-gray-400">No prediction yet.</p>
        )}
      </div>
    </div>
  )
}
