'use client'

import { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import { User, signInWithPopup, signOut, onAuthStateChanged } from 'firebase/auth'
import { doc, setDoc, getDoc, serverTimestamp } from 'firebase/firestore'
import { auth, googleProvider, db, COLLECTIONS } from '@/lib/firebase'

interface AuthContextType {
  user: User | null
  loading: boolean
  signInWithGoogle: () => Promise<void>
  logout: () => Promise<void>
  userProfile: UserProfile | null
}

interface UserProfile {
  uid: string
  email: string
  displayName: string
  photoURL: string
  assignedModel: string | null
  totalEvaluations: number
  createdAt: Date
  lastActiveAt: Date
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      setUser(user)

      if (user) {
        // Get or create user profile in Firestore
        const userRef = doc(db, COLLECTIONS.USERS, user.uid)
        const userSnap = await getDoc(userRef)

        if (userSnap.exists()) {
          const data = userSnap.data()
          setUserProfile({
            uid: user.uid,
            email: user.email || '',
            displayName: user.displayName || '',
            photoURL: user.photoURL || '',
            assignedModel: data.assignedModel || null,
            totalEvaluations: data.totalEvaluations || 0,
            createdAt: data.createdAt?.toDate() || new Date(),
            lastActiveAt: new Date()
          })

          // Update last active
          await setDoc(userRef, { lastActiveAt: serverTimestamp() }, { merge: true })
        } else {
          // Create new user profile
          const newProfile = {
            uid: user.uid,
            email: user.email || '',
            displayName: user.displayName || '',
            photoURL: user.photoURL || '',
            assignedModel: null,
            totalEvaluations: 0,
            createdAt: serverTimestamp(),
            lastActiveAt: serverTimestamp()
          }
          await setDoc(userRef, newProfile)
          setUserProfile({
            ...newProfile,
            createdAt: new Date(),
            lastActiveAt: new Date()
          } as UserProfile)
        }
      } else {
        setUserProfile(null)
      }

      setLoading(false)
    })

    return () => unsubscribe()
  }, [])

  const signInWithGoogle = async () => {
    try {
      await signInWithPopup(auth, googleProvider)
    } catch (error) {
      console.error('Error signing in with Google:', error)
      throw error
    }
  }

  const logout = async () => {
    try {
      await signOut(auth)
    } catch (error) {
      console.error('Error signing out:', error)
      throw error
    }
  }

  return (
    <AuthContext.Provider value={{ user, loading, signInWithGoogle, logout, userProfile }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
