import React, { useState } from 'react'
import axios from 'axios'

export default function Login(){
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    try{
      const data = new URLSearchParams({ username: email, password, grant_type: '' })
      const res = await axios.post('/api/auth/token', data)
      localStorage.setItem('token', res.data.access_token)
      setMessage('Logged in!')
    }catch(err){
      setMessage('Login failed')
    }
  }

  return (
    <div className="max-w-md mx-auto bg-white/5 p-8 rounded-xl border border-white/10">
      <h2 className="text-2xl font-bold mb-4">Admin/Doctor Login</h2>
      <form onSubmit={submit} className="space-y-4">
        <input className="w-full px-4 py-2 rounded bg-black/40 border border-white/10" placeholder="Email" value={email} onChange={e=>setEmail(e.target.value)} />
        <input className="w-full px-4 py-2 rounded bg-black/40 border border-white/10" type="password" placeholder="Password" value={password} onChange={e=>setPassword(e.target.value)} />
        <button className="w-full py-2 rounded bg-primary hover:bg-primary/80 font-semibold">Login</button>
      </form>
      {message && <p className="mt-3 text-sm text-gray-300">{message}</p>}
    </div>
  )
}
