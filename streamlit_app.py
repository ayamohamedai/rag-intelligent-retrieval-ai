import React, { useState } from 'react';
import { Upload, Search, FileText, MessageCircle, Zap, Globe } from 'lucide-react';

const RAGSystem = () => {
  const [documents, setDocuments] = useState([]);
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files);
    const processedFiles = files.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: (file.size / 1024).toFixed(1),
      type: file.type,
      content: `محتوى تجريبي من ملف ${file.name}. هذا النص يمثل المحتوى المستخرج من الملف المرفوع.`
    }));
    
    setDocuments(prev => [...prev, ...processedFiles]);
  };

  const handleSearch = () => {
    if (!query.trim()) return;
    
    setIsProcessing(true);
    
    // محاكاة معالجة الاستعلام
    setTimeout(() => {
      let response = '';
      
      if (query.toLowerCase().includes('مرحبا') || query.toLowerCase().includes('hello')) {
        response = `🤖 **مرحباً بك في نظام RAG العالمي!**

**إليك ما يمكنني مساعدتك فيه:**
- 📚 تحليل المستندات المرفوعة
- 🔍 البحث في محتوى الملفات
- 💡 الإجابة على أسئلتك بذكاء
- 🌍 دعم اللغة العربية والإنجليزية

**المستندات المتاحة:** ${documents.length} ملف
**حالة النظام:** ✅ جاهز للعمل

كيف يمكنني مساعدتك اليوم؟`;
      } else if (documents.length === 0) {
        response = `❌ **لا توجد مستندات مرفوعة**

الرجاء رفع بعض المستندات أولاً لأتمكن من الإجابة على استعلامك: "${query}"

**الأنواع المدعومة:**
- PDF 📄
- DOCX 📝  
- TXT 📋`;
      } else {
        response = `🎯 **استعلامك:** ${query}

📖 **النتيجة من المستندات:**

بناءً على تحليل ${documents.length} مستند مرفوع، وجدت المعلومات التالية:

**📄 من الملف: ${documents[0]?.name}**
النص ذو الصلة: "${documents[0]?.content.substring(0, 150)}..."
درجة التطابق: 0.87

**💡 الملخص:**
تم العثور على معلومات ذات صلة بسؤالك في المستندات المرفوعة. النظام قام بتحليل المحتوى وتقديم أفضل إجابة ممكنة.

**🔗 المراجع:** ${documents.length} مستند`;
      }
      
      setAnswer(response);
      setIsProcessing(false);
    }, 1500);
  };

  const clearAll = () => {
    setDocuments([]);
    setQuery('');
    setAnswer('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white p-6">
      {/* Header */}
      <div className="max-w-6xl mx-auto">
        <div className="bg-gradient-to-r from-purple-600 to-pink-600 p-8 rounded-2xl mb-8 text-center shadow-2xl">
          <div className="flex items-center justify-center mb-4">
            <Globe className="w-12 h-12 mr-4 animate-spin" />
            <h1 className="text-4xl font-bold">النظام العالمي RAG</h1>
          </div>
          <p className="text-xl opacity-90">Intelligent Retrieval & Generation</p>
          <p className="text-lg mt-2 opacity-80">🚀 استرجاع المستندات + توليد الإجابات</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="bg-gradient-to-br from-pink-500 to-red-500 p-6 rounded-2xl shadow-xl">
            <div className="flex items-center mb-4">
              <Upload className="w-8 h-8 mr-3" />
              <h2 className="text-2xl font-bold">📤 ارفع مستنداتك</h2>
            </div>
            
            <div className="border-2 border-dashed border-white border-opacity-50 rounded-xl p-8 text-center mb-4 hover:border-opacity-100 transition-all duration-300">
              <input
                type="file"
                multiple
                accept=".pdf,.docx,.txt"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                <FileText className="w-16 h-16 mx-auto mb-4 opacity-70" />
                <p className="text-xl mb-2">Drag and drop files here</p>
                <p className="text-sm opacity-75">Limit 200MB per file • PDF, DOCX, TXT</p>
                <button className="bg-white text-purple-600 px-6 py-2 rounded-full font-bold mt-4 hover:bg-opacity-90 transition-all">
                  Browse files
                </button>
              </label>
            </div>

            {/* Uploaded Files */}
            {documents.length > 0 && (
              <div className="space-y-3">
                <h3 className="font-bold text-lg">📁 الملفات المرفوعة:</h3>
                {documents.slice(0, 3).map((doc) => (
                  <div key={doc.id} className="bg-white bg-opacity-20 p-3 rounded-lg">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{doc.name}</span>
                      <span className="text-sm opacity-75">{doc.size} KB</span>
                    </div>
                  </div>
                ))}
                {documents.length > 3 && (
                  <p className="text-center opacity-75">+{documents.length - 3} ملف آخر...</p>
                )}
              </div>
            )}
          </div>

          {/* Query Section */}
          <div className="bg-gradient-to-br from-blue-500 to-cyan-500 p-6 rounded-2xl shadow-xl">
            <div className="flex items-center mb-4">
              <MessageCircle className="w-8 h-8 mr-3" />
              <h2 className="text-2xl font-bold">💡 اطرح سؤالك</h2>
            </div>

            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="اكتب سؤالك هنا..."
              className="w-full h-32 p-4 rounded-xl bg-white bg-opacity-20 placeholder-white placeholder-opacity-70 text-white resize-none border-0 focus:outline-none focus:ring-4 focus:ring-white focus:ring-opacity-30"
              dir="rtl"
            />

            <div className="flex gap-3 mt-4">
              <button
                onClick={handleSearch}
                disabled={isProcessing}
                className="flex-1 bg-gradient-to-r from-yellow-400 to-orange-500 text-white py-3 rounded-xl font-bold hover:shadow-lg transform hover:scale-105 transition-all duration-200 flex items-center justify-center disabled:opacity-50"
              >
                {isProcessing ? (
                  <Zap className="w-5 h-5 mr-2 animate-pulse" />
                ) : (
                  <Search className="w-5 h-5 mr-2" />
                )}
                {isProcessing ? 'جاري المعالجة...' : '🔍 البحث والإجابة'}
              </button>
              
              <button
                onClick={clearAll}
                className="bg-red-500 hover:bg-red-600 px-6 py-3 rounded-xl font-bold transition-all"
              >
                مسح
              </button>
            </div>

            {/* System Stats */}
            <div className="mt-4 bg-white bg-opacity-20 p-3 rounded-xl">
              <div className="grid grid-cols-2 gap-4 text-center">
                <div>
                  <div className="text-2xl font-bold">{documents.length}</div>
                  <div className="text-sm opacity-75">📚 مستندات</div>
                </div>
                <div>
                  <div className="text-2xl font-bold">✅</div>
                  <div className="text-sm opacity-75">جاهز</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Answer Section */}
        {answer && (
          <div className="mt-8 bg-gradient-to-br from-green-500 to-teal-500 p-6 rounded-2xl shadow-xl">
            <h2 className="text-2xl font-bold mb-4 flex items-center">
              <Zap className="w-8 h-8 mr-3" />
              ✨ الإجابة
            </h2>
            <div className="bg-white bg-opacity-20 p-6 rounded-xl">
              <div className="whitespace-pre-line text-right leading-relaxed">
                {answer}
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="mt-12 text-center opacity-70">
          <p>🤖 نظام RAG العالمي - تم تطويره باستخدام React & AI</p>
        </div>
      </div>
    </div>
  );
};

export default RAGSystem;
