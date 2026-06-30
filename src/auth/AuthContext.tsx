import React, { useState, useEffect, useCallback, useContext, createContext } from 'react';

export type AppUser = {
  uid: string;
  email: string | null;
  displayName: string | null;
};

interface AuthContextType {
  user: AppUser | null;
  loading: boolean;
  signInWithGoogle: () => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

const DEFAULT_USER: AppUser = {
  uid: 'local-user',
  email: 'user@localhost',
  displayName: 'Local User',
};

const STORAGE_KEY = 'jarvis_user';

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<AppUser | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    let resolved: AppUser = DEFAULT_USER;
    if (saved) {
      try {
        resolved = JSON.parse(saved) as AppUser;
      } catch {
        // corrupted entry — fall back to default
      }
    }
    setUser(resolved);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(resolved));
    setLoading(false);
  }, []);

  const handleSignInWithGoogle = useCallback(async () => {
    setUser(DEFAULT_USER);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(DEFAULT_USER));
  }, []);

  const handleSignOut = useCallback(async () => {
    localStorage.removeItem(STORAGE_KEY);
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        signInWithGoogle: handleSignInWithGoogle,
        signOut: handleSignOut,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};
