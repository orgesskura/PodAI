import axios from 'axios';

const API_URL = 'http://localhost:3001/api'; // Adjust this URL to match your backend

export const login = async (email, password) => {
  try {
    const response = await axios.post(`${API_URL}/login`, { email, password });
    const { token } = response.data;
    localStorage.setItem('token', token);
    return { success: true, token };
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Login failed');
  }
};

export const register = async (email, password) => {
  try {
    const response = await axios.post(`${API_URL}/register`, { email, password });
    const { token } = response.data;
    localStorage.setItem('token', token);
    return { success: true, token };
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Registration failed');
  }
};

export const logout = async () => {
  try {
    await axios.post(`${API_URL}/logout`, {}, {
      headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
    });
    localStorage.removeItem('token');
    return { success: true };
  } catch (error) {
    throw new Error(error.response?.data?.message || 'Logout failed');
  }
};