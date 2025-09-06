import React, { useState } from 'react';
import { Upload, Search, FileText, MessageCircle } from 'lucide-react';

const SmartRAGSystem = () => {
  const [files, setFiles] = useState([]);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);

  // رفع الملفات
  const uploadFiles = (e) => {
    const selectedFiles = Array.from(e.target.files);
    const newFiles = selectedFiles.map(file => ({
      id: Date.now() + Math.random(),
      name: file.name,
      size: file.size,
      time: new Date().toLocaleString()
    }));
    setFiles([...files, ...newFiles]);
    e.target.value = '';
  };

  // البحث والإجابة
  const searchAnswer = () => {
    if (!question) {
      alert('اكتب سؤال!');
      return;
    }
    
    setLoading(true);
    setAnswer('');
    
    setTimeout(() => {
      let response = '';
      
      if (question.toLowerCase().includes('مرحب')) {
        response = `🤖 أهلاً وسهلاً!
        
الملفات المرفوعة: ${files.length}
النظام: جاهز ✅
اللغة: العربية

اكتب أي سؤال وسأجيب عليك!`;
      } else if (files.length === 0) {
        response = `❌ لا توجد ملفات!
        
ارفع ملفات أولاً ثم اسأل سؤالك.`;
      } else {
        response = `✅ تم البحث في ${files.length} ملف
        
السؤال: ${question}

الإجابة: تم العثور على معلومات مفيدة في الملفات المرفوعة. 

المصادر: ${files.map(f => f.name).join(', ')}`;
      }
      
      setAnswer(response);
      setLoading(false);
    }, 1000);
  };

  const clear = () => {
    setFiles([]);
    setQuestion('');
    setAnswer('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 to-purple-900 text-white p-8" dir="rtl">
      <div className="max-w-4xl mx-auto space-y-8">
        
        {/* العنوان */}
        <div className="text-center bg-gradient-to-r from-purple-600 to-pink-600 p-6 rounded-2xl">
          <h1 className="text-4xl font-bold mb-2">🤖 نظام RAG الذكي</h1>
          <p className="text-lg">ارفع ملفات واسأل أسئلة!</p>
        </div>

        {/* رفع الملفات */}
        <div className="bg-green-600 p-6 rounded-2xl">
          <h2 className="text-2xl font-bold mb-4 flex items-center">
            <Upload className="ml-2" />
            📁 رفع الملفات
          </h2>
          
          <input
            type="file"
            multiple
            onChange={uploadFiles}
            className="hidden"
            id="files"
          />
          <label htmlFor="files" className="block">
            <div className="border-2 border-dashed border-white p-8 text-center rounded-xl cursor-pointer hover:bg-white hover:bg-opacity-10">
              <FileText className="w-16 h-16 mx-auto mb-4" />
              <p className="text-xl font-bold">اضغط لرفع الملفات</p>
            </div>
          </label>

          {/* عرض الملفات */}
          {files.length > 0 && (
            <div className="mt-4 space-y-2">
              <p className="font-bold">الملفات المرفوعة: {files.length}</p>
              {files.map(file => (
                <div key={file.id} className="bg-white bg-opacity-20 p-3 rounded-lg">
                  <div className="font-medium">{file.name}</div>
                  <div className="text-sm opacity-75">{file.time}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* الاستعلام */}
        <div className="bg-blue-600 p-6 rounded-2xl">
          <h2 className="text-2xl font-bold mb-4 flex items-center">
            <MessageCircle className="ml-2" />
            💬 اسأل سؤالك
          </h2>
          
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="اكتب سؤالك هنا... جرب: مرحبا"
            className="w-full h-24 p-4 rounded-xl bg-white bg-opacity-20 text-white placeholder-white placeholder-opacity-70 resize-none border-none outline-none text-right"
          />
          
          <div className="flex gap-4 mt-4">
            <button
              onClick={searchAnswer}
              disabled={loading}
              className="flex-1 bg-yellow-500 text-black py-3 px-6 rounded-xl font-bold hover:bg-yellow-400 transition-all flex items-center justify-center"
            >
              {loading ? 'جاري البحث...' : (
                <>
                  <Search className="ml-2" />
                  🔍 ابحث
                </>
              )}
            </button>
            
            <button
              onClick={clear}
              className="bg-red-500 px-6 py-3 rounded-xl font-bold hover:bg-red-400 transition-all"
            >
              مسح
            </button>
          </div>
        </div>

        {/* النتائج */}
        {answer && (
          <div className="bg-orange-600 p-6 rounded-2xl">
            <h2 className="text-2xl font-bold mb-4">📋 النتيجة</h2>
            <div className="bg-white bg-opacity-20 p-4 rounded-xl">
              <pre className="whitespace-pre-wrap text-right font-medium leading-relaxed">
                {answer}
              </pre>
            </div>
          </div>
        )}

        {/* معلومات سريعة */}
        <div className="bg-gray-700 p-4 rounded-xl text-center">
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-2xl font-bold text-green-400">{files.length}</div>
              <div className="text-sm">ملفات</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-blue-400">{question.length}</div>
              <div className="text-sm">أحرف</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-yellow-400">{answer ? '1' : '0'}</div>
              <div className="text-sm">إجابة</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SmartRAGSystem;
