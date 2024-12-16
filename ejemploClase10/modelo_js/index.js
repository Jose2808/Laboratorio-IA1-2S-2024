import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as readline from 'readline'

const modelJson = JSON.parse(fs.readFileSync('D:\\monte\\OneDrive\\Escritorio\\Universidad\\Practicas Finales\\Inteligencia Artificial\\Laboratorio-IA1-2S-2024\\ejemploClase10\\modelo_js\\model.json'))
const tokenizerJson = JSON.parse(fs.readFileSync('D:\\monte\\OneDrive\\Escritorio\\Universidad\\Practicas Finales\\Inteligencia Artificial\\Laboratorio-IA1-2S-2024\\ejemploClase10\\modelo_js\\tokenizer.json'))

const wordIndex = JSON.parse(tokenizerJson['config']['word_index'])

//const model = await tf.loadLayersModel('http://127.0.0.1:8080/model.json')
const model = await tf.loadLayersModel(tf.io.fromMemory(modelJson))
console.log(model.summary())

function textsToSequences(texts){
    return texts.map(text => {
        const words = text.toLowerCase().trim().split(" ")

        return words.map(word => wordIndex[word] || 0)
    })
}

function padSequences(sequences, maxLength, paddingType = 'pre', truncatingType = 'pre', paddingValue = 0){
    return sequences.map(seq => {
        if (seq.length > maxLength) {
            if(truncatingType === 'pre'){
                seq = seq.slice(seq.length - maxLength)
            }else {
                seq = seq.slice(0, maxLength)
            }
        } 

        if (seq.length < maxLength){
            const paddingLength = maxLength - seq.length
            const paddingArray = new Array(paddingLength).fill(paddingValue)

            if(paddingType === 'pre') {
                seq = [...paddingArray, ...seq]
            }else{
                seq = [...seq, ...paddingArray]
            }
        }

        return seq
    })
}

const r1 = readline.createInterface({
    input: process.stdin,
    output: process.stdout
}) 

const askQuestion = (question) => {
    return new Promise((resolve) => {
        r1.question(question, (answer) => {
            resolve(answer)
        })
    })
}

const main = async () => {
    let user_input = await askQuestion("Hello ")
    user_input = user_input.toLowerCase()

    let sequences = textsToSequences([user_input])
    console.log(sequences)

    sequences = padSequences(sequences, 5., 'pre', 'pre', 0)
    console.log(sequences)
}

main()