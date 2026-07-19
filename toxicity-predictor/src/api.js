import axios from 'axios';

export const API_URL =
  process.env.REACT_APP_API_URL || 'https://toxicity-predictor-2310.onrender.com';

const apiClient = axios.create({
  baseURL: API_URL,
  timeout: 120000,
  headers: {
    Accept: 'application/json',
    'Content-Type': 'application/json'
  }
});

export const predictToxicity = async ({ smiles, compoundName }) => {
  const response = await apiClient.post('/api/predict', {
    smiles,
    compound_name: compoundName
  });

  return response.data;
};

export const batchPredictToxicity = async ({ smilesList, compoundNames }) => {
  const response = await apiClient.post('/api/batch-predict', {
    smiles_list: smilesList,
    compound_names: compoundNames
  });

  return response.data;
};

export const checkHealth = async () => {
  const response = await apiClient.get('/api/health');
  return response.data;
};

const hasJsonParseError = (error) =>
  error instanceof SyntaxError ||
  /Unexpected token|JSON|invalid json/i.test(error?.message || '');

export const formatApiError = (error) => {
  if (axios.isAxiosError(error)) {
    if (error.code === 'ECONNABORTED') {
      return 'The request timed out while contacting the backend. Please try again.';
    }

    if (error.response) {
      const payload = error.response.data;
      const backendMessage =
        payload?.error ||
        payload?.message ||
        payload?.detail ||
        payload?.title;

      if (backendMessage) {
        return backendMessage;
      }

      if (hasJsonParseError(error.cause)) {
        return 'The backend returned an invalid response. Please try again.';
      }

      return `Server error (${error.response.status}). Please try again.`;
    }

    if (error.request) {
      return 'Cannot reach the backend service. Please try again in a moment.';
    }
  }

  if (hasJsonParseError(error)) {
    return 'The backend returned an invalid response. Please try again.';
  }

  return error?.message || 'An unexpected error occurred while contacting the backend.';
};