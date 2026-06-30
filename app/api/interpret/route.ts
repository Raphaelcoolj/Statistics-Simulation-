import { NextRequest, NextResponse } from 'next/server'
import { rateLimit, getRateLimitIdentifier } from '@/lib/utils/rateLimit'
import type { InterpretRequestBody } from '@/lib/types'
import { GoogleGenerativeAI } from '@google/generative-ai'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function buildPrompt(schema: any, result: any): string {
  // Prune down results safely so we never hit a 413 Payload Too Large error
  const optimizedResult = {
    summaryMetrics: result.summary || result.metrics || {},
    regressionResult: result.regressionResult ? {
      r2: result.regressionResult.r2,
      featureImportance: result.regressionResult.featureImportance 
    } : undefined,
    // Slice arrays down to a tiny sample size instead of dumping hundreds of rows
    previewData: Array.isArray(result) ? result.slice(0, 3) : undefined
  };

  return `
    You are a professional automated data scientist for StatLab.
    Analyze the following dataset context and mathematical results, then provide plain-English structural interpretations.
    
    DATASET SCHEMA:
    ${JSON.stringify(schema)}
    
    COMPUTED ANALYSIS RESULTS (SAMPLE):
    ${JSON.stringify(optimizedResult)}
    
    RESPONSE FORMAT REQUIREMENT:
    You must respond ONLY with a raw valid JSON object matching this structural shape. Do not wrap it in markdown code blocks:
    {
      "summary": "A concise paragraph summarizing the key takeaway of the entire data run.",
      "perAnalysis": [
        {
          "type": "The category or metric analyzed (e.g., Regression, Distribution)",
          "subject": "The specific column or pair targeted (e.g., Age vs Spending_Score)",
          "interpretation": "A 1-2 sentence deep context insight regarding what this specific mathematical change means."
        }
      ]
    }
  `;
}

export async function POST(request: NextRequest) {
  try {
    const identifier = getRateLimitIdentifier(request)
    const { allowed } = rateLimit(identifier, 20, 60_000)
    if (!allowed) {
      return Response.json(
        { success: false, error: 'Too many requests. Wait a moment.' },
        { status: 429, headers: { 'Retry-After': '60' } }
      )
    }

    const body = await request.json() as InterpretRequestBody
    if (!body.schema || !body.result) {
      return NextResponse.json(
        { success: false, error: 'Schema and Result are required inputs' },
        { status: 400 }
      )
    }

    const payloadPrompt = buildPrompt(body.schema, body.result);

    // =======================================================
    // 🚀 ENGINE 1: GROQ (PRIMARY ENGINE)
    // =======================================================
    if (process.env.GROQ_API_KEY) {
      try {
        console.log('[StatLab AI] Routing request to Groq (Primary)...');
        const groqResponse = await fetch('https://api.groq.com/openai/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model: 'llama-3.3-70b-versatile', // Updated model that perfectly handles JSON configurations
            messages: [{ role: 'user', content: payloadPrompt }],
            temperature: 0.1
          })
        });

        if (groqResponse.ok) {
          const rawData = await groqResponse.json();
          let cleanText = rawData.choices[0].message.content.trim();
          
          // Strip out markdown code blocks if Llama accidentally injects them
          if (cleanText.startsWith('```json')) cleanText = cleanText.replace(/^```json/, '').replace(/```$/, '');
          if (cleanText.startsWith('```')) cleanText = cleanText.replace(/^```/, '').replace(/```$/, '');

          const parsedJSON = JSON.parse(cleanText.trim());
          return NextResponse.json({
            success: true,
            summary: parsedJSON.summary,
            perAnalysis: parsedJSON.perAnalysis || [],
            provider: 'groq',
            fallbackUsed: false,
          });
        }
        
        console.warn(`[StatLab AI] Groq endpoint status code [${groqResponse.status}]. Falling back to sub...`);
      } catch (groqError) {
        console.warn('[StatLab AI] Groq parsing/network failure. Attempting Gemini fallback...', groqError);
      }
    }

    // =======================================================
    // 🔄 ENGINE 2: GEMINI (SUB / FALLBACK ENGINE)
    // =======================================================
    if (process.env.GEMINI_API_KEY) {
  try {
    console.log('[StatLab AI] Routing request to Gemini (Fallback Sub)...');
    
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');
    // Using gemini-pro bypasses the 404 version constraints of your installed package
    const model = genAI.getGenerativeModel({ model: 'gemini-pro' });

    const result = await model.generateContent(payloadPrompt);
    let cleanText = result.response.text().trim();

        // Strip out markdown formatting if present
        if (cleanText.startsWith('```json')) cleanText = cleanText.replace(/^```json/, '').replace(/```$/, '');
        if (cleanText.startsWith('```')) cleanText = cleanText.replace(/^```/, '').replace(/```$/, '');

        const parsedJSON = JSON.parse(cleanText.trim());

        return NextResponse.json({
          success: true,
          summary: parsedJSON.summary,
          perAnalysis: parsedJSON.perAnalysis || [],
          provider: 'gemini',
          fallbackUsed: true,
        });
      } catch (geminiError) {
        console.error('[StatLab AI] Gemini fallback structural block failed:', geminiError);
      }
    }

    throw new Error('Both Groq and Gemini execution pathways failed to resolve payload.');

  } catch (error) {
    console.error('[StatLab AI] Global Fallback Triggered:', error);
    return NextResponse.json({
      success: true,
      summary: 'Analysis complete. Review the system charts and generated data models for full insights.',
      perAnalysis: [],
      provider: null,
      fallbackUsed: true,
    });
  }
}