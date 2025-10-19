import React, { useEffect, useState } from 'react'
import api from '../lib/api'

export default function Admin(){
  const [users, setUsers] = useState([])
  const [preds, setPreds] = useState([])

  useEffect(()=>{
    (async()=>{
      try{
        const [u,p] = await Promise.all([
          api.get('/api/users'),
          api.get('/api/predictions')
        ])
        setUsers(u.data)
        setPreds(p.data)
      }catch(e){
        // ignore
      }
    })()
  },[])

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-bold mb-3">Users</h2>
        <div className="overflow-auto rounded border border-white/10">
          <table className="w-full text-sm">
            <thead className="bg-white/5">
              <tr>
                <th className="text-left p-2">ID</th>
                <th className="text-left p-2">Email</th>
                <th className="text-left p-2">Role</th>
              </tr>
            </thead>
            <tbody>
              {users.map(u=> (
                <tr key={u.id} className="odd:bg-white/5">
                  <td className="p-2">{u.id}</td>
                  <td className="p-2">{u.email}</td>
                  <td className="p-2">{u.role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section>
        <h2 className="text-2xl font-bold mb-3">Recent Predictions</h2>
        <div className="overflow-auto rounded border border-white/10">
          <table className="w-full text-sm">
            <thead className="bg-white/5">
              <tr>
                <th className="text-left p-2">ID</th>
                <th className="text-left p-2">User</th>
                <th className="text-left p-2">Model</th>
                <th className="text-left p-2">Label</th>
                <th className="text-left p-2">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {preds.map(p=> (
                <tr key={p.id} className="odd:bg-white/5">
                  <td className="p-2">{p.id}</td>
                  <td className="p-2">{p.user_id || '-'}{/* not in schema */}</td>
                  <td className="p-2">{p.model_type}</td>
                  <td className="p-2">{p.predicted_label}</td>
                  <td className="p-2">{(p.confidence*100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
