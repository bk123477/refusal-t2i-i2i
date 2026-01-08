// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getAuth } from "firebase/auth";

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyCYZNH-D5KEPc_2f1Gr9vcmFpyd29t97xU",
  authDomain: "acrb-e8cb4.firebaseapp.com",
  projectId: "acrb-e8cb4",
  storageBucket: "acrb-e8cb4.firebasestorage.app",
  messagingSenderId: "87810362498",
  appId: "1:87810362498:web:9a13c8f15886030ab6b89b"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Firebase services
export const db = getFirestore(app);
export const auth = getAuth(app);

export default app;
