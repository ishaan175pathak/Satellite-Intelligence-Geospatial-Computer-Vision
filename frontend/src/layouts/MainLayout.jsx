import React from 'react';
import Navbar from '../components/Navbar';
import UserInput from '../components/UserInput';

const MainLayout = () => {
  return (
    <div className="relative">
      <Navbar />
      <UserInput />      
    </div>
  );
};

export default MainLayout;