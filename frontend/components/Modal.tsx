"use client";

import { useEffect, useState } from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
}

export default function Modal({ isOpen, onClose, title, children }: ModalProps) {
  const [show, setShow] = useState(isOpen);

  useEffect(() => {
    setShow(isOpen);
  }, [isOpen]);

  if (!show) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center overflow-x-hidden overflow-y-auto outline-none focus:outline-none bg-black/50 backdrop-blur-sm transition-all">
      <div className="relative w-full max-w-md mx-auto my-6 px-4">
        <div className="relative flex flex-col w-full bg-white dark:bg-gray-800 border-0 rounded-2xl shadow-2xl outline-none focus:outline-none transform transition-all scale-100 animate-fade-in-up">
          {/* Header */}
          <div className="flex items-center justify-between p-5 border-b border-solid border-gray-100 dark:border-gray-700 rounded-t">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white">
              {title || "알림"}
            </h3>
            <button
              className="p-1 ml-auto bg-transparent border-0 text-gray-400 hover:text-gray-900 dark:hover:text-white float-right text-3xl leading-none font-semibold outline-none focus:outline-none transition-colors"
              onClick={onClose}
            >
              <span className="block w-6 h-6 text-2xl leading-none">
                ×
              </span>
            </button>
          </div>
          {/* Body */}
          <div className="relative p-6 flex-auto">
            <p className="my-2 text-gray-600 dark:text-gray-300 text-base leading-relaxed">
              {children}
            </p>
          </div>
          {/* Footer */}
          <div className="flex items-center justify-end p-5 border-t border-solid border-gray-100 dark:border-gray-700 rounded-b">
            <button
              className="bg-indigo-600 text-white active:bg-indigo-700 font-bold uppercase text-sm px-6 py-3 rounded-xl shadow hover:shadow-lg outline-none focus:outline-none mr-1 mb-1 ease-linear transition-all duration-150"
              type="button"
              onClick={onClose}
            >
              확인
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
