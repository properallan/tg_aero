%% =============================
%%      IMPORTANTE
%% ESTE ARQUIVO DEVE ESTAR SALVO COMO
%%      ISO - 8859 - 1
%% =============================

%% +++++++++++++++++++++++++++++
%% ufabcFHZh
%% é uma edição de FHZ, orientado por Prof. Magno e com contribuição de Reginaldo Cardoso e tradução de Guilherme Rinaldo.
%% É a classe para gerar os arquivos de dissertações e teses com os padrões usados pela UFABC.
%%
%% Histórico de versões:
%%
%% >>>>> ufabcFHZd
%%
%% Adicionado links coloridos
%% Seções Comentadas identificando funções
%% 10.05.2015
%% Descoberto erro em de \frontmatter em \makefrontpagex - FOLHA DE ASSINATURAS
%% 11.05.2015
%% Corrigida estrutura de date - day - month - year
%% Adicionando trechos corretos de:
%% 	ufabcThiagoRevisaoAssinatura
%%		FOLHA DE ASSINATURAS - Erro encontrado e isolado
%%		FICHA DE REVISÃO - Concluida
%%
%% >>>>> ufabcFHZe
%%
%% 19.05.2015
%% Testando geração de classe com nome correto
%% Utilizando controle de edições com letras do alfabeto para permitir
%% 		uso de mesmo nome na classe do arquivo e Replace fácil no futuro.
%% Versões passadas apenas renomeadas
%% 22.05.2015
%% Aprendido e explicado como inserir lista de símbolos e abreviações.
%% 		Técnicas citadas nas seções de:
%%			Relatorio_TCC_Include_Int_1_0_FHZ.tex
%% Adicionado
%% 		\chapter*{\large\textbf{Resumo - Abstract}}
%% 		Resumo e Abstract agora possuem a mesma formatação dos demais \chapter{}.
%% 27.05.2015
%% Lista de assinaturas corrigida e concluída.
%%
%% >>>>> ufabcFHZf
%%
%% 12.11.2015
%% Atualizando posições de acordo com novo guia de normalização ufabc.
%% Correção da posição da numeração de todos os ambientes
%% Criação dos ambientes agradecimentos e epigrafe para padronizar
%% Uso melhorado de Lista de Símbolos e Abreviaturas. Descobrindo opção para editar entrada de dados.
%%
%% >>>>> ufabcFHZg
%%
%% 17.02.2016
%% Criado ambiente citacaoDireta
%% 		Para uso de citação diretas de mais de 3 linhas, com recuo de texto.
%%
%% >>>>> ufabcFHZh
%%
%% 27.10.2016
%% Alteração de todos os caracteres acentuados que não sejam comentários
%% 09.11.2016
%% Criando mecanismos para troca de idioma entre português e inglês automaticamente com opção declarada da classe. Baseado na edição feita por  e com colaboração de Guilerme Rinaldo.
%% 08.12.2016
%% Corrigindo posição do texto da folha de rosto
%% 11.12.2016
%% Alteração de comando para ficar mais claro
%% 		\makefrontpagex -> \makepagesignatures
%% Estudando como deixar o arquivo perfeito para imprimir frente e verso de uma só vez
%% Pré-textual: somente de um lado
%% Textual e pós: ambos os lados
%%
%% Problema: Fazer isso nas listas de figuras e similares
%%
%% Alterado coppe.ist para remoção ``p.''
%% delim_0   ", p. " % BATATA
%% Alterada marcação do ``p.'' nas listas e sumário
%%
%% Alterado ``coppe.ist'' para conter ``\thispagestyle{empty}'' no ambiente
%% 		glossary que ele gera e corrigir a numeração que aparecia e ninguém via
%%
%% Resultado obtido com sucesso usando ``\usepackage{everyshi}''
%% 	E escrevendo seus valores dentro do lof.
%%
%% 21.12.2016
%% Opção VF quase totalmente concluída
%% Organizando o código
%% Problema pendente é o ``AtBeginDocument{...}'' para o ``Everyshipout'' dentro do ``iftoggle{toggleVersaoFinal}''
%%
%% 23.12.2016
%% Opção VF concluída.
%% Atualizando a versão EN com trabalho de Guilherme.
%%
%%
%%
%% +++++++++++++++++++++++++++++

%% This is file `ufabcFHZh.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% ufabcFHZh.dtx  (with options: `class')
%%
%% This is the `ufabcFHZh' document class used for composition of LaTeX
%% monographs (master and PhD thesis, and PhD qualifying exams)
%% satisfying the requirements of the Federeal University of ABC, Santo André, Brazil.
%%
%% Original files
%%
%% Copyright (C) 2008 Vicente Helano and George Ainsworth Jr.
%%
%% This program is free software; you can redistribute it and/or modify
%% it under the terms of the GNU General Public License version 3 as
%% published by the Free Software Foundation.
%%
%% This program is distributed in the hope that it will be useful,
%% but WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%% GNU General Public License version 3 for more details.
%%
%% You should have received a copy of the GNU General Public License
%% version 3 along with this package (see COPYING file).
%% If not, see <http://www.gnu.org/licenses/>.
%%
%% Original :
%% Author(s): Vicente Helano,
%%            George Ainsworth
%%
%% ufabcFHZh:
%% Author(s): FHZ - B. Eng. Fernando Henrique G. Zucatelli
%% 			  Prof. Dr. Magno Enrique Mendoza Meza
%% Collaboration(s):
%% 			  B.Eng. Reginaldo Cardoso
%% Translator(s):
%% 			  B. Eng. Guilherme Rinaldo - BATATA - Condicional
%% =====================================================

%% ============ Definições e Classes - Início
\def\filename{ufabcFHZh.dtx}
\def\fileversion{v2.0}
\def\filedate{2016/12/12} 	 % aaaa / mm / dd
\NeedsTeXFormat{LaTeX2e}[1995/12/01] % aaaa / mm / dd
\ProvidesClass{ufabcFHZh}[\filedate\ \fileversion\ COPPE-UFABC-FHZ Dissertations and Thesis]
\ProvidesPackage{ufabcFHZh}[\filedate\space\fileversion]
\LoadClass[12pt,a4paper,twoside]{book}
\RequirePackage[sort&compress]{natbib}
\RequirePackage{hyphenat}
\RequirePackage{lastpage}
\RequirePackage{ifthen}
\RequirePackage{graphicx}
\RequirePackage{setspace}
\RequirePackage{tabularx}
\RequirePackage{eqparbox}
\RequirePackage{ltxcmds}
\RequirePackage{etoolbox} 	% Permite o uso de \newtoggle para criar desvios de fluxo
\RequirePackage{afterpage} % Controla o comportamente da próxima página dentro de um ambiente - Usado nos ambientes abstract e foreignabstract
% --- O uso do pacote {babel} é selecionado através da variável ``toggleENTrue''
%\RequirePackage[english]{babel}
%\RequirePackage[english,brazil]{babel}
% --- 
\RequirePackage{everyshi} % Necessário para lof, lot, lab, los, toc serem impressos somente do lado ímpar/direito.
%% ============ Definições e Classes - Fim

% *************** Pacotes e formatos - Inicio
% --- Margens
\RequirePackage[a4paper,bindingoffset=0.0cm,vcentering=true,%
top=3.0cm,bottom=2.0cm,left=3.0cm,right=2.0cm]{geometry}
% --- Não faço ideia do que seja,não vou arriscar mexer - FHZ 21.12.2016
\def\CoppeTeX{{\rm C\kern-.05em{\sc o\kern-.025em p\kern-.025em
			p\kern-.025em e}}\kern-.08em
	T\kern-.1667em\lower.5ex\hbox{E}\kern-.125emX\spacefactor1000}
% ---
% *************** Pacotes e formatos - Fim

% -- Controle de formatação versão final e rascunho econômico - Início
\newtoggle{toggleVersaoFinal} 	% Opção para selecionar com \iftoggle
\togglefalse{toggleVersaoFinal} % Padrão é falso
\DeclareOption{VF}{%
	\toggletrue{toggleVersaoFinal} % Verdadeiro: ``VF'' vem da opção da classe
}
% -- Controle de formatação versão final e rascunho econômico - Fim

%% ============ Código de formatação somente de um lado para impressão lado a lado da versão final das listas - Início
% BATATA - Projetar Toogle com Declare Option para selecionar comportamento para qualificação e defesa economizando página e para a impressão da versão final. Ou ter a opção de duas classes com nomes diferentes... mas ter duas classes não agrada, gera duplicata de código e isso é quase certamente ruim. 12.12.2016.
% BATATA - 21.12.2016: Funciona sem o ``\iftoggle'', mas eu preciso selecionar o que fazer de acordo com esse toggle.
% BATATA - Tentando inverter o ``\AtBeginDocument'' com ``\iftoggle''
% BATATA - 21.12.2016 - FUNCIONA!!!
\AtBeginDocument{%
	% Referência: http://tex.stackexchange.com/questions/118170/table-of-content-only-on-right-side-pages - Boris
	%\usepackage{everyshi}
	%\makeatletter
	%\EveryShipout{\if@mainmatter\else\ifodd\c@page\null\pagebreak\fi\fi}
	% Essa sequência escreve o comando \EveryShipout{} dentro da lista de figura (lof) e propaga suas características para as demais, tabela (lot), símbolos (los) e abreviações (lab).
	%\AtBeginDocument{%
	%	% Lista de figuras - e consequência tabelas, símbolos e abreviaturas
	%	\addtocontents{lof}{\protect\makeatletter
	%		\protect\EveryShipout{%
	%			\protect\if@mainmatter
	%			\protect\else
	%			\protect\ifodd
	%			\protect\c@page\null\protect\pagebreak
	%			\protect\fi
	%			\protect\fi
	%		}
	%	}
	%}
	% Resultado que aparece no arquivo ``.lof'':
	% \iftoggle {toggleVersaoFinal}{\EveryShipout {\if@mainmatter \else \ifodd \c@page \hbox {}\pagebreak \fi \fi } } {} 
	% Lista de figuras - e consequência tabelas, símbolos e abreviaturas; e sumário.
	\addtocontents{lof}{\protect\makeatletter
		\protect\iftoggle{toggleVersaoFinal}{%
			\protect\EveryShipout{%
				\protect\if@mainmatter
				\protect\else
				\protect\ifodd
				\protect\c@page\null\protect\pagebreak
				\protect\fi
				\protect\fi
			}
		}
		{}
	}
}
%% ============ Código de formatação somente de um lado para impressão lado a lado da versão final das listas - Fim

%%%%%%%%%%%%%%% ---------- Textos comuns independente do idioma - Início
\newcommand{\local@universityname}{Universidade Federal do ABC}
\newcommand{\foreign@universityname}{Federal University of ABC}
\newcommand{\@UniversidadeAbrev}{UFABC}
\newcommand\local@deptstring{Programa}
\newcommand\foreign@deptstring{Department}
\newcommand\local@cityname{Santo Andr\'{e}}
\newcommand\local@statename{SP}
\newcommand\local@countryname{Brasil}
\newcommand\foreign@countryname{Brazil}
\newcommand{\@Endereco}{Av. dos Estados, 5001}
\newcommand{\@CEP}{CEP 09210-580}
\newcommand{\@EnderecoCompletoUniversidade}{{\bf\local@universityname}, \@Endereco -- \@CEP -- \local@cityname -- \local@statename}
\newcommand{\@Telefone}{Tel. 0 XX 11 -- 4996 -- 0086}
\newcommand{\@Site}{www.ufabc.edu.br}
\newcommand{\@TelefoneSite}{{\bf\@Telefone} -- {\bf\@Site}}
% ---
\newcommand{\local@advisorstring}{Orientador}
\newcommand{\local@coadvisorstring}{Coorientador}
\newcommand{\foreign@advisorstring}{Advisor}
\newcommand{\foreign@coadvisorstring}{Coadvisor}
\newcommand{\@pdfsubject}{\local@doctype\ de \@degreename\ em \local@deptname\ da POSMEC/\@UniversidadeAbrev}
\newcommand{\@AnoPrefixoNum}{20}
% --- Palavras-chaves E keywords
\newcommand{\@Resumo}{Resumo}
\newcommand{\@Abstract}{Abstract}
\newcommand{\@stringPalavraschave}{\textbf{Palavras-chave:}}
\newcommand{\@stringKeywords}{\textbf{Keywords:}}
%%%%%%%%%%%%%%% ---------- Textos comuns independente do idioma - Fim

%%%% ++++++++++++++++ Criação das variáveis de controle de fluxo - Início
% -- Variável de controle PT-EN
\newtoggle{toggleENTrue} 	% Opção para selecionar com \iftoggle

% --- Variável de controle do tipo de documento
\newboolean{maledoc}
\setboolean{maledoc}{false}
%%%% ++++++++++++++++ Criação das variáveis de controle de fluxo - Fim

% BATATA MESTRA - Alterações de idioma com o Guilherme Rinaldo
% *************** Idioma - Alteração de textos - Inicio
%%
%% Todos comandos iniciando com ``\@''
%% @: Usado para evitar que o usuário crie um comando com mesmo nome e sobre-escreva o comando do .cls
%%
%% \newcommand para criar a variável em PT e \renewcommand para modificar caso seja escolhido em inglês
%%
%% -- // -- // -- // --
%\DeclareOption{PT}{%
% -- PT => EN <- false
\togglefalse{toggleENTrue}
% --
\newcommand{\@FigName}{Figura}
\newcommand{\@TabName}{Tabela}
\newcommand{\@CursoPosGradEm}{PÓS-GRADUA\c{C}\~{A}O EM}
\newcommand{\@Cursoposgradem}{Pós-Gradua\c{c}\~{a}o em}
\newcommand{\@PosGradEm}{PÓS-GRADUA\c{C}\~{A}O EM}
\newcommand{\@FOLHAASSINATURAS}{FOLHA DE ASSINATURAS}
\newcommand{\@ExemplarRevisado}{Este exemplar foi revisado e alterado em rela\c{c}\~{a}o \`{a} vers\~{a}o original, de acordo com as observa\c{c}\~{o}es levantadas pela banca no dia da defesa, sob responsabilidade \'{u}nica do autor e com a anu\^{e}ncia de seu orientador.}
% ----
\newcommand{\@Agradecimentos}{Agradecimentos}
\newcommand{\@Epigrafe}{Ep\'{\i}grafe}
% ----
\newcommand{\@AssinaturaOrientador}{Assinatura do orientador:}
\newcommand{\@AssinaturaAutor}{Assinatura do autor:}
\newcommand{\@AoCurso}{ao Curso de Pós-gradua{\c c}{\~ a}o da}
\newcommand{\@ComoParteRequisitos}{como parte dos requisitos necess{\'a}rios {\`a} obten{\c c}{\~a}o do}
\newcommand{\@graude}{grau de}
\newcommand{\@em}{em}
\newcommand{\@de}{de}
\newcommand{\@strTitulo}{T{\' i}tulo}
% -----
\newcommand{\@ListaIlustracoes}{Lista de Ilustra{\c c}{\~o}es}
\newcommand{\glossaryname}{Gloss\'{a}rio} 
\newcommand{\listabbreviationname}{Lista de Abreviaturas}
\newcommand{\listsymbolname}{Lista de S{\' i}mbolos}
\newcommand{\@BANCAEXAMINADORA}{BANCA EXAMINADORA}
\newcommand{\@AssinaturasMembrosBancaExaminado}{Assinaturas dos membros da Banca Examinadora que avaliou e aprovou a Defesa de}
\newcommand{\@candidato}{candidato}
% ---- Texto de Aprovação
\newcommand{\@TextoAprovacao}{%
	{\@AssinaturasMembrosBancaExaminado} {\local@doctype}
	de {\MakeLowercase\@degreename} 
	do {\@candidato} \nohyphens{{\@authname} {\@authsurn}} {\@em} {\ifnum\number\day<10 0\fi}\number\day\ de \local@monthname\ de \number\year
}
% ---- Texto Apresentação Folha de Rosto
\newcommand{\@TextoApresentacaoFolhaDeRosto}{%
	{\local@doctype}
	{\ifthenelse{\boolean{maledoc}}{apresentado}{apresentada}}
	\sloppy\nohyphens{{\@AoCurso} \local@universityname, {\@ComoParteRequisitos} {\@graude} \local@degname\ {\@em} {\local@deptname.\par}}
}
%%}

%% -- // -- // -- // --
\DeclareOption{EN}{%
	% -- EN => EN <- true
	\toggletrue{toggleENTrue}
	% --
	\renewcommand{\@FigName}{Figure}
	\renewcommand{\@TabName}{Table}
	\renewcommand{\@CursoPosGradEm}{GRADUATE COURSE IN}
	\renewcommand{\@Cursoposgradem}{Graduate Course in}
	\renewcommand{\@PosGradEm}{GRADUATE COURSE IN}
	\renewcommand{\@FOLHAASSINATURAS}{SIGN SHEET}
	\renewcommand{\@ExemplarRevisado}{This exemplar was revised and altered from the original version according to the observations done by the defense board at the defense day, under responsability of author and advisor's consent.}
	% ----
	\renewcommand{\@Agradecimentos}{Acknowledgement}
	\renewcommand{\@Epigrafe}{Epigraph}
	% ----
	\renewcommand{\@AssinaturaOrientador}{Advisor's signature:}
	\renewcommand{\@AssinaturaAutor}{Author's signature:}
	\renewcommand{\@AoCurso}{to the Graduate course of}
	\renewcommand{\@ComoParteRequisitos}{as a partial requirement for obtaining a Master's degree in} 
	\renewcommand{\@em}{in}
	\renewcommand{\@de}{of}
	\renewcommand{\@strTitulo}{Title}
	% ----
	\renewcommand{\@ListaIlustracoes}{List of Figures}
	% BATATA
	\renewcommand{\glossaryname}{Glossary} 
	\renewcommand{\listabbreviationname}{List of Abreviations}
	\renewcommand{\listsymbolname}{List of Symbols}
	% BATATA
	\renewcommand{\@BANCAEXAMINADORA}{EXAMINING BOARD}
	\renewcommand{\@AssinaturasMembrosBancaExaminado}{Signatures of examining board members, which examinated and approved  the defense of}
	\renewcommand{\@candidato}{candidate}
	% ---- Texto de Aprovação
	\renewcommand{\@TextoAprovacao}{%
		{\@AssinaturasMembrosBancaExaminado} {\foreign@doctype}
		{\MakeLowercase\@degreename} 
		{\@candidato} \nohyphens{{\@authname} {\@authsurn}} {\@em} {\foreign@monthname} {\ifnum\number\day<10 0\fi}{\number\day}%
		\ifthenelse{\equal{\number\day}{1}\or\equal{\number\day}{21}\or\equal{\number\day}{31}}{%
			\textsuperscript{st}}{%
			\ifthenelse{\equal{\number\day}{2}\or\equal{\number\day}{22}}{\textsuperscript{nd}}{%
				\ifthenelse{\equal{\number\day}{3}\or\equal{\number\day}{23}}{\textsuperscript{rd}}{%
					\textsuperscript{th}%
				}%
			}%
		}%
		, \number\year
	}
	% ---- Texto Apresentação Folha de Rosto
	\renewcommand{\@TextoApresentacaoFolhaDeRosto}{%
		{\local@doctype} 
		{presented}
		\sloppy\nohyphens{{\@AoCurso} \foreign@universityname, {\@ComoParteRequisitos} {\foreign@deptname.\par}}
	}
}
% *************** Idioma - Alteração de textos - Fim

%% ---------------->>>>>>> Edição de listas - Inicio
%%% ---- >>> Exemplo
% http://tex.stackexchange.com/questions/97484/how-to-change-the-font-of-page-numbers-in-toc-lot-lof-without-tocloft

%\let\oldl@chapter\l@chapter
%\def\l@chapter#1#2{\oldl@chapter{#1}{\textcolor{red}{\itshape#2}}}

%%% ---- >>> Execução
% --- Figura
\let\oldl@figure\l@figure
\def\l@figure#1#2{\oldl@figure{{\@FigName} #1}{#2}}
% --- Tabela
\let\oldl@table\l@table
\def\l@table#1#2{\oldl@table{{\@TabName} #1}{#2}}
%% --- Número capítulo
%\let\oldl@chapter\l@chapter
%\def\l@chapter#1#2{\oldl@chapter{#1}{p. #2}}
%% --- Número seções
%\let\old@dottedcontentsline\@dottedtocline
%\def\@dottedtocline#1#2#3#4#5{%
%	\old@dottedcontentsline{#1}{#2}{#3}{#4}{p. #5}}
%% ---------------->>>>>>> Edição de listas - Fim

% *************** Títulos - Inicio
\DeclareOption{msc}{%
  \newcommand{\@degree}{M.Sc.}
  \newcommand{\@degreename}{MASTER}
  \newcommand{\local@degname}{Mestre}
  \newcommand{\foreign@degname}{Master}
  \newcommand\local@doctype{Dissertação}
  \newcommand\foreign@doctype{Dissertação}
}
\DeclareOption{dscexam}{%
  \newcommand{\@degree}{D.Sc.}
  \newcommand{\@degreename}{Doutorado}
  \newcommand{\local@degname}{Doutor}
  \newcommand{\foreign@degname}{Doctor}
  \setboolean{maledoc}{true}
  \newcommand\local@doctype{Exame de Qualifica{\c c}{\~ a}o}
  \newcommand\foreign@doctype{Qualifying Exam}
}
\DeclareOption{dsc}{%
  \newcommand{\@degree}{D.Sc.}
  \newcommand{\@degreename}{Doutorado}
  \newcommand{\local@degname}{Doutor}
  \newcommand{\foreign@degname}{Doctor}
  \newcommand\local@doctype{Tese}
  \newcommand\foreign@doctype{Thesis}
}
% *************** Títulos - Fim

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\palavrachave}[1]{#1}
\newcommand{\keywordeng}[1]{#1}
\newcounter{pagenumber_frontmatter}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\DeclareOption{numbers}{%
  \PassOptionsToPackage{numbers}{natbib}
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\onehalfspacing
\DeclareOption{doublespacing}{%
  \doublespacing
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ProcessOptions\relax
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% BATATA - Seleção - funciona somente com toogle
%% -*-*-*-*-*-*-*- Seleção de idioma do pacote babel - Inicio
\iftoggle{toggleENTrue}{%
	\RequirePackage[english,english]{babel}
}{%
	\RequirePackage[english,brazil]{babel}
}
%% -*-*-*-*-*-*-*- Seleção de idioma do pacote babel - Fim

% *************** Cursos - Inicio
\newcommand\department[1]{%
  \ifthenelse{\equal{#1}{PEB}}
    {\global\def\local@deptname{Engenharia Biom{\' e}dica}
     \global\def\foreign@deptname{Biomedical Engineering}}{}
  \ifthenelse{\equal{#1}{PEC}}
    {\global\def\local@deptname{Engenharia Civil}
     \global\def\foreign@deptname{Civil Engineering}}{}
  \ifthenelse{\equal{#1}{PEE}}
    {\global\def\local@deptname{Engenharia El{\' e}trica}
     \global\def\foreign@deptname{Electrical Engineering}}{}
  \ifthenelse{\equal{#1}{PEM}}
    {\global\def\local@deptname{Engenharia Mec{\^ a}nica}
     \global\def\foreign@deptname{Mechanical Engineering}}{}
  \ifthenelse{\equal{#1}{PEMM}}
    {\global\def\local@deptname{Engenharia Metal{\' u}rgica e de Materiais}
\global\def\foreign@deptname{Metallurgical and Materials Engineering}}{}
  \ifthenelse{\equal{#1}{PEN}}
    {\global\def\local@deptname{Engenharia Nuclear}
     \global\def\foreign@deptname{Nuclear Engineering}}{}
  \ifthenelse{\equal{#1}{PENO}}
    {\global\def\local@deptname{Engenharia Oce{\^ a}nica}
     \global\def\foreign@deptname{Ocean Engineering}}{}
  \ifthenelse{\equal{#1}{PPE}}
    {\global\def\local@deptname{Planejamento Energ{\' e}tico}
     \global\def\foreign@deptname{Energy Planning}}{}
  \ifthenelse{\equal{#1}{PEP}}
    {\global\def\local@deptname{Engenharia de Produ{\c c}{\~ a}o}
     \global\def\foreign@deptname{Production Engineering}}{}
  \ifthenelse{\equal{#1}{PEQ}}
    {\global\def\local@deptname{Engenharia Qu{\' i}mica}
     \global\def\foreign@deptname{Chemical Engineering}}{}
  \ifthenelse{\equal{#1}{PESC}}
    {\global\def\local@deptname{Engenharia de Sistemas e Computa{\c c}{\~ a}o}
     \global\def\foreign@deptname{Systems Engineering and Computer Science}}{}
  \ifthenelse{\equal{#1}{PET}}
    {\global\def\local@deptname{Engenharia de Transportes}
     \global\def\foreign@deptname{Transportation Engineering}}{}
    %  ===== Adicionado em 18.11.2015 - fonte: http://propg.ufabc.edu.br/cursos/
  \ifthenelse{\equal{#1}{BIOS}}
	{\global\def\local@deptname{Biossistemas}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{BIOT}}
    {\global\def\local@deptname{Biotecnoci{\^ e}ncia}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{CCM}}
  {\global\def\local@deptname{Ci{\^ e}ncia da Computa{\c c}{\~ a}o}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{CTA}}
    {\global\def\local@deptname{Ci{\^ e}ncia e Tecnologia Ambiental}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{CTQ}}
    {\global\def\local@deptname{Ci{\^ e}ncia e Tecnologia/Qu{\' i}mica}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{CHS}}
    {\global\def\local@deptname{Ci{\^ e}ncias Humanas e Sociais}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{EEN}}
    {\global\def\local@deptname{Energia}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{MEB}}
    {\global\def\local@deptname{Engenharia Biom{\' e}dica}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{MEI}}
    {\global\def\local@deptname{Engenharia da Informa{\c c}{\~ a}o}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{EGI}}
    {\global\def\local@deptname{Engenharia de Instrumenta{\c c}{\~ a}o, Automa{\c c}{\~ a}o e Rob{\' o}tica}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{EHFCM}}
    {\global\def\local@deptname{Ensino, Hist{\' o}ria e Filosofia das Ci{\^ e}ncias e Matem{\' a}tica}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{EDI}}
    {\global\def\local@deptname{Evolução e Diversidade }
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{FIL}}
    {\global\def\local@deptname{Filosofia}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{FIS}}
  {\global\def\local@deptname{F{\' i}sica}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{MAT}}
  {\global\def\local@deptname{Matem{\' a}tica}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{NMA}}
  {\global\def\local@deptname{Nanoci{\^ e}ncias e Materiais Avançados}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{NCG}}
  {\global\def\local@deptname{Neuroci{\^ e}ncia e Cogni{\c c}{\~ a}o}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{PGT}}
  {\global\def\local@deptname{Planejamento e Gestão do Territ{\' o}rio}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{PPU}}
  {\global\def\local@deptname{Pol{\' i}ticas P{\' u}blicas}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  %% Mestrado Profissional
  \ifthenelse{\equal{#1}{PROFMAT}}
    {\global\def\local@deptname{Mestrado Profissional em Matem{\' a}tica em Rede Nacional}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  \ifthenelse{\equal{#1}{MNPEF}}
    {\global\def\local@deptname{Mestrado Nacional Profissional em Ensino de F{\' i}sica}
    \global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  %% Doutorado Acadêmico Industrial
  \ifthenelse{\equal{#1}{DAI}}
  {\global\def\local@deptname{Doutorado Acad{\^ e}mico Industrial}
  	\global\def\foreign@deptname{NOT DEFINED - edit ufabcFHZ.cls}}{}
  %%  Adicionado em 25.08.2019 - Mestrado de Gestão da Inovação
  \ifthenelse{\equal{#1}{INV}}
  {\global\def\local@deptname{Engenharia e Gestão da Inovação}
  	\global\def\foreign@deptname{Engenharia e Gestão da Inovação}}{}
}
% *************** Cursos - Fim

%%%%%%%%%%%%%%%%%% ----- Nomes - Início
\renewcommand\title[1]{%
  \global\def\local@title{#1}%
}
\newcommand\foreigntitle[1]{%
  \global\def\foreign@title{#1}%
}
\newcount\@advisor\@advisor0
\newcommand\advisor[4]{%
  \global\@namedef{CoppeAdvisorTitle:\expandafter\the\@advisor}{#1}
  \global\@namedef{CoppeAdvisorName:\expandafter\the\@advisor}{#2}
  \global\@namedef{CoppeAdvisorSurname:\expandafter\the\@advisor}{#3}
  \global\@namedef{CoppeAdvisorDegree:\expandafter\the\@advisor}{#4}
  \global\advance\@advisor by 1
  % BATATA - Somente um orientador e o segundo é o coorientador
  % ======= Código antigo - Início
	%  \ifnum\@advisor>1
	%    \renewcommand\local@advisorstring{Orientadores}
	%    \renewcommand\local@coadvisorstring{Coorientadores}
	%    \renewcommand\foreign@advisorstring{Advisors}
	%    \renewcommand\foreign@coadvisorstring{Coadvisors}
	%  \fi
  % ======= Código antigo - Fim
}
\newcount\@examiner\@examiner0
\newcommand\examiner[3]{%
  \global\@namedef{CoppeExaminer:\expandafter\the\@examiner}{#1\ #2 - #3}
  \global\advance\@examiner by 1
}

\newcount\@coordinator\@coordinator0
\newcommand\coordinator[3]{%
  \global\@namedef{CoppeCoordinator:\expandafter\the\@coordinator}{#1\ #2, #3}
  \global\advance\@coordinator by 1
}

\renewcommand\author[2]{%
  \global\def\@authname{#1}
  \global\def\@authsurn{#2}
}
\renewcommand\date[3]{%
  \day=#1
  \month=#2
  \year=#3
}
\newcommand\local@monthname{\ifcase\month\or
  Janeiro\or Fevereiro\or Mar{\c c}o\or Abril\or Maio\or Junho\or
  Julho\or Agosto\or Setembro\or Outubro\or Novembro\or Dezembro\fi}
\newcommand\foreign@monthname{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi}
\newcounter{keywords}
\newcommand\keyword[1]{%
  \global\@namedef{CoppeKeyword:\expandafter\the\c@keywords}{#1}
  \global\addtocounter{keywords}{1}
}
\def\endkeywords{\par}
%%%%%%%%%%%%%%%%%% ----- Nomes - Fim

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%% ********** Geração de elementos pré-textuais - Inicio
\renewcommand\frontmatter{%
	% -- Abertura direita na Versão Final, qualquer uma na rascunho - Início
	\@toogleOpenRightTrueOrFalse % Comando personalizado chamado em dois locais
	% -- Abertura direita na Versão Final, qualquer uma na rascunho - Fim
	
	%% //// Pular folha em relação à Capa - Versão Final
	\iftoggle{toggleVersaoFinal}{\cleardoublepage}{\clearpage}
	\@mainmatterfalse
	
	%% //// Gerar Folha de Rosto
	%\pagenumbering{roman}
	%\thispagestyle{empty}
	%\pagestyle{empty}
	\setcounter{page}{1} % BATATA - Primeira folha a contar é a de rosto
	\makefrontpage
	
	%% //// Gerar Ficha Catalográfica
	\clearpage % Ficha Catalográfica sempre no verso da Folha de Rosto 
	%\pagestyle{empty}
	\ifthenelse{\boolean{maledoc}}{}{\makecatalog}%
	
	%% //// Gerar Ficha de Revisão
	\clearpage % Ficha de Revisão sempre após Ficha Catalográfica
	%\pagestyle{empty}
	\ifthenelse{\boolean{maledoc}}{}{\makerevisao}%
	
	%% //// Gerar Folha de Assinaturas
	\iftoggle{toggleVersaoFinal}{\cleardoublepage}{\clearpage}
	\pagestyle{empty}
	\ifthenelse{\boolean{maledoc}}{}{\makepagesignatures}
}
%%%%%%%%%%% ********** Geração de elementos pré-textuais - Fim

%% //// Gerar Ficha de Aprovação - Inicio - ``\frontmatterx'' não é chamado no código
\newcommand\frontmatterx{%
	\iftoggle{toggleVersaoFinal}{\cleardoublepage}{\clearpage}
	\@mainmatterfalse
	% Não é necessário exibir numeração no pré-texto
	\pagenumbering{roman}
	\thispagestyle{empty}
	\makepagesignatures
	\pagestyle{empty}
	\ifthenelse{{maledoc}}{}{\makecatalog}
}
%% //// Gerar Ficha de Aprovação - Fim

%%%%%%%%%%% ********** Geração de elementos textuais - Inicio
\renewcommand\mainmatter{%
	\ufabcFHZh@mainBegin
	\iftoggle{toggleVersaoFinal}{\cleardoublepage}{\clearpage}
	% -- Abertura direita na Versão Final, qualquer uma na rascunho - Início
	\@toogleOpenRightTrueOrFalse % Comando personalizado chamado em dois locais
	% -- Abertura direita na Versão Final, qualquer uma na rascunho - Fim
	\@mainmattertrue
	\makeatletter \let\ps@plain\ps@myheadings \makeatother
	\thispagestyle{empty}
	\pagestyle{myheadings}
	\pagenumbering{arabic}
}
%%%%%%%%%%% ********** Geração de elementos textuais - Fim

%%%%%%%%%%% ********** Geração de elementos pós-textuais - Inicio
\renewcommand\backmatter{%
	\if@openright \cleardoublepage \else \clearpage \fi
}
%%%%%%%%%%% ********** Geração de elementos pós-textuais - Fim

%%%%% ========================= Capa - Inicio
\renewcommand\maketitle{%
	\thispagestyle{empty}
	\pagestyle{empty}
	%\pagenumbering{alph}
	\ltx@ifpackageloaded{hyperref}{\ufabcFHZh@hypersetup}{}%
	\begin{titlepage}
		\begin{center} 
			\vspace*{1.5mm}
			{\MakeUppercase{%
					\iftoggle{toggleENTrue}{\foreign@universityname}{\local@universityname}}} \\ 
			{\@CursoPosGradEm} {\MakeUppercase
				{\iftoggle{toggleENTrue}{\foreign@deptname}{\local@deptname}}
			}
		\end{center} 
		\vfill
		\begin{center}
			\nohyphens{{\@authname} {\@authsurn}}\par
			%  \vspace*{3cm}
			\vfill
			\nohyphens{\MakeUppercase\local@title}\par
		\end{center}
		\vspace*{\fill}
%		\begin{center}
%			\iftoggle{toggleENTrue}{\foreign@doctype}{\local@doctype} % BATATA
%		\end{center}
		
		\vspace*{\fill}
		\begin{center}
			%\local@cityname\par
			{\local@cityname, {\local@statename} -- 
				{\iftoggle{toggleENTrue}{\foreign@countryname}{\local@countryname}}
			}\par
			{\iftoggle{toggleENTrue}{{\foreign@monthname}}{{\local@monthname} de }} {\number\year}  % BATATA - Correção padrão de DATA da CAPA
		\end{center}
	\end{titlepage}
	
	\global\let\maketitle\relax%
	\global\let\and\relax
}
%%%%% ========================= Capa - Fim

%%%%% ========================= Folha de Rosto - Inicio
\newcommand\makefrontpage{%
	\begin{center}
		%    \sloppy\nohyphens{\MakeUppercase\local@title}\par
		{{\@authname} {\@authsurn}}\par
		\vfill
		\sloppy\nohyphens{\MakeUppercase\local@title}\par
		\vfill
		%\iftoggle{toggleENTrue}{\foreign@doctype}{\local@doctype}\par %Não é necessário escrever o tipo de arquivo BATATA{\local@doctype}
	\end{center}\par
	
	\vfill
	\frontpage@maintext % Definido a seguir
	\vfill
	
	%  \noindent\local@approvedname:\par %%%% Examinada
	\hspace{3.5cm}
	\begin{minipage}{.7\textwidth}
		\begin{flushleft}
			\begin{tabularx}{10.45cm}[b]{@{}l@{}>{\raggedright\arraybackslash}X@{}}
				\@escritaNomesOrientadorCoorientador
			\end{tabularx}\par
		\end{flushleft}
	\end{minipage}
	\vspace*{\fill}
	\frontpage@bottomtext % Definido a seguir
}
%%%%% ========================= Folha de Rosto - Fim
%%%%% ------- Texto Folha de Rosto - Parte dos Requisitos - Inicio
\newcommand\frontpage@maintext{
	\hspace{3.5cm}
	\begin{minipage}{.7\textwidth}
		\noindent 
		\@TextoApresentacaoFolhaDeRosto
%		{\local@doctype}
%		{\iftoggle{toggleENTrue}{presented}{\ifthenelse{\boolean{maledoc}}{apresentado}{apresentada}}}
%		 % BATATA - Revisar para inglês - pendente
%		\sloppy\nohyphens{{\@AoCurso} \local@universityname, {\@ComoParteRequisitos} {\@graude} \local@degname\ {\@em} {\local@deptname.\par}}
	\end{minipage}
}
%%%%% ------- Texto Folha de Rosto - Parte dos Requisitos - Fim
%%%%% ------- Texto Linha Final Folha de Rosto - Inicio
\newcommand\frontpage@bottomtext{%
	\begin{center}
		{{\local@cityname, {\local@statename} -- 
				{\iftoggle{toggleENTrue}{\foreign@countryname}{\local@countryname}}
		}}\par
		{\iftoggle{toggleENTrue}{{\foreign@monthname}}{{\local@monthname} de }} {\number\year}
	\end{center}%
}
%%%%% ------- Texto Linha Final Folha de Rosto - Fim

%%%%% ========================= Ficha Catalográfica - Inicio
\newcommand\makecatalog{%
	%% 12.12.2016
	%% Ficha Catalográfica Corrigida para ficar semelhante ao do site da biblioteca
	%% Fontes: http://portal.biblioteca.ufabc.edu.br/index.php?option=com_content&view=article&id=51&Itemid=27
	%% http://portal.biblioteca.ufabc.edu.br/ficha_catalografica/
	\vspace*{\fill}
	\begin{center}
		% BATATA - Isso não pode estar aqui na versão liberada - Início
		%Gerado automaticamente pelo próprio modelo {\LaTeX} da POSMEC porque o FHZ é foda. E não ia ficar perdendo tempo preenchendo dados da ficha da biblioteca.
		% BATATA - Isso não pode estar aqui na versão liberada - Fim
		\vspace*{1cm}
		\setlength{\fboxsep}{5mm}
		\framebox[145mm][c]{\makebox[10mm][c]{}%
			\begin{minipage}[c]{125mm}
				\small
				%\footnotesize
				%\tiny
				\setlength{\parindent}{5mm}
				\noindent\sloppy\nohyphens\@authsurn,
				\nohyphens\@authname\par
				\nohyphens{{\local@title}/{\@authname} {\@authsurn}. -- {\number\year}.}\par
				%\pageref{front:pageno},
				\pageref{LastPage}
				p.\@ifundefined{r@cat:lofflag}{}{\pageref{cat:lofflag}}\par
				%% não é necessário informar o tamanho da folha: $29,7$cm.
				\nohyphens{%
					\begin{tabularx}{120mm}[b]{@{}l@{ }>{\raggedright\arraybackslash}X@{}}
						\@escritaNomesOrientadorCoorientador % BATATA - Código de geração em separado
				\end{tabularx}}\par
				\nohyphens{
					{\iftoggle{toggleENTrue}{\foreign@doctype}{\local@doctype}} ({\@degreename}) -- %{\MakeLowercase\@degreename}
					{\iftoggle{toggleENTrue}{\foreign@universityname}{\local@universityname}}, {\@Cursoposgradem}
					{\iftoggle{toggleENTrue}{\foreign@deptname}{\local@deptname}}, {\local@cityname}, {\number\year}.}\par 
				%\bibname: p. \pageref{bib:begin} -- \pageref{bib:end}.\par % BATATA - Não é necessário indicar as páginas onde estão as referências biliográficas
				\count1=0
				\count2=1
				\nohyphens{\@whilenum \count1<\value{keywords} \do {%
						\number\count2. \csname CoppeKeyword:\the\count1 \endcsname.
						\advance\count1 by 1
						\advance\count2 by 1}
					I. \csname CoppeAdvisorSurname:0\endcsname, \csname CoppeAdvisorName:0\endcsname%
					\ifthenelse{\@advisor>1}
					{\, II. {\csname CoppeAdvisorSurname:1\endcsname, \csname CoppeAdvisorName:1\endcsname}
						III. {\@Cursoposgradem} {\iftoggle{toggleENTrue}{\foreign@deptname}{\local@deptname}}, {\number\year}.
						IV. {\@strTitulo}}
					{\, II. {\@Cursoposgradem} {\iftoggle{toggleENTrue}{\foreign@deptname}{\local@deptname}}, {\number\year}.
						III. {T{\' i}tulo}.}
				}
				% BATATA - Esse ``T{\' i}tulo'' não faz sentido - seria o título do trabalho? Se for o título ``{\local@title}''
				% BATATA - Explicação: http://edcapistrano.blogspot.com.br/2011/03/fazendo-uma-ficha-catalografica.html
		\end{minipage}}
	\end{center}
	%\vspace*{\fill}}
	\vspace*{1cm}
}
%%%%% ========================= Ficha Catalográfica - Fim

%%%%% ========================= Ficha de Revisão - Início
\newcommand\makerevisao{%
	%\vspace*{\fill}
	\begin{center}
		\setlength{\fboxsep}{5mm}
		\framebox[0.8\linewidth][c]{\makebox[5mm][c]{}%
			\begin{minipage}[l]{0.75\linewidth}
				\begin{sloppypar} % BATATA - Revisão
					{\@ExemplarRevisado}
				\end{sloppypar}
				\vspace*{0.75cm}
				
				% *********** Editado FHZ \line - Inicio
				%\line(x-slope,y-slope){length}
				\noindent \local@cityname,  % BATATA - Correção de padrão de data
				{\iftoggle{toggleENTrue}
					{\line(1,0){110} \; \line(1,0){40} \;,}
					{\line(1,0){40} \; de \line(1,0){110} \; de}
				}
				{\@AnoPrefixoNum} \line(1,0){30}
				\vspace*{0.75cm}
				
				\noindent {\@AssinaturaAutor} % BATATA - assina
				\line(1,0){220}
				\vspace*{0.75cm}
				
				\noindent{\@AssinaturaOrientador} % BATATA - assina
				\line(1,0){220}
				% *********** Editado FHZ \line - Fim
		\end{minipage}}
	\end{center}
	\vspace*{\fill}
}
%%%%% ========================= Ficha de Revisão - Fim

%%%%% ========================= Ficha de Aprovação - Início
\newcommand\makepagesignatures{%
	\begin{center}
		\vspace*{-3.cm}
		\begin{figure}[!htb]
			\begin{minipage}[b]{0.2\linewidth}
				\includegraphics[width=\linewidth]{Textuais/Images/ufabclogo.png}
			\end{minipage} \hfill
			\vspace*{-1.5cm}
			\begin{minipage}[b]{0.70\linewidth}
				\centering
				\normalsize {\@PosGradEm} {\MakeUppercase\local@deptname}
			\end{minipage}
		\end{figure}
	\end{center}\par
	%  \vspace*{10mm}
	\vfill
	\vspace*{-1.5cm}
	\begin{center}
		{\@FOLHAASSINATURAS} \\ \vspace*{-0.55cm}
		{\hrulefill}
	\end{center}%\par
	\frontpage@ambienteTextAprovacao
	\begin{flushleft} %%%% Banca examinadora
		\count1=0
		\@whilenum \count1<\@examiner \do {
			\vskip52\p@ \baselineskip12pt
			%\eqparbox{signture@list}
			{\hrulefill}\\
			%\eqparbox{signture@list}
			{\protect%\centering%
				\  \csname CoppeExaminer:\the\count1 \endcsname\ \ \ }\\
			\advance\count1 by 1}
	\end{flushleft}
	\par
	\vfill
	\begin{center}
		{\tiny \@EnderecoCompletoUniversidade \\ \@TelefoneSite}
	\end{center}
}
%%%%% ========================= Ficha de Aprovação - Fim
%%%%% ------- Texto Folha de aprovação - Inicio
\newcommand\frontpage@ambienteTextAprovacao{%
	\begin{flushleft} % *** Editado FHZ
		\noindent 
		{\@TextoAprovacao}:
	\end{flushleft}
}
%%%%% ------- Texto Folha de aprovação - Fim

%%%%% ========================= Dedicatória - Inicio
\newcommand\dedication[1]{%
	\gdef\@dedic{#1}
	\iftoggle{toggleVersaoFinal}{\cleardoublepage}{\clearpage}
	\vspace*{\fill}
	\begin{flushright}
		\begin{minipage}{60mm}
			\raggedleft \textit \normalsize \@dedic
		\end{minipage}
	\end{flushright}
}
%%%%% ========================= Dedicatória - Fim

%%%%%%%%%%%%%%%% - Novos ambientes com  nome de capítulo definido - Início
%%%%% ========================= Título do Agradecimentos - Inicio
\newenvironment{agradecimentos}{% 
	\chapter*{{\@Agradecimentos}} % *************** Editado FHZ 
	\vspace{-12ex}
	\thispagestyle{empty}
	\pagestyle{empty}
}
%%%%% ========================= Título do Agradecimentos  - Fim

%%%%% ========================= Título do Epígrafe - Inicio
\newenvironment{epigrafe}{%
	%\clearpage % BATATA - impressão
	\chapter*{{{\@Epigrafe}}} % *************** Editado FHZ
	\vspace{-12ex}
	\thispagestyle{empty}
	\pagestyle{empty}
}
%%%%% ========================= Título do Epígrafe  - Fim

%%%%% ========================= Título do Resumo - Inicio
\newenvironment{abstract}{%
	% BATATA
	% % % % % % % % % % % % % % % % % %
	%\usepackage{titling}
	%\RequirePackage{titling}
	%\posttitle{\par\end{center}\vskip 0.5em}
	%\setlength{\droptitle}{-10pt}
	% Referência: http://tex.stackexchange.com/questions/591/removing-vertical-space-inside-maketitle
	% % % % % % % % % % % % % % % % % %
	%\usepackage{titlesec} % http://ctan.org/pkg/titlesec
	%\titlespacing{\chapter}{-10pt}{0pt}{<after-sep>}
	% Referência: http://tex.stackexchange.com/questions/11233/distance-between-chapter-title-and-text
	% % % % % % % % % % % % % % % % % %
	%\addtocontents{abstract}{\vskip-80pt}
	%\addtocontents{abstract}{\vspace{-10ex}}
	%\addtocontents{toc}{\vskip-40pt}
	% Referência: http://tex.stackexchange.com/questions/13319/fncychap-package-vertical-gap-space-between-text-and-heading-too-large-how-t
	% http://tex.stackexchange.com/questions/13351/fncychap-vertical-space-between-text-and-heading-too-large-shift-left-heading
	% % % % % % % % % % % % % % % % % % -- Esta solução deve ser inserida no "..._Input_Pacotes.tex"
	%\makeatletter
	%\renewcommand*{\@makechapterhead}[1]{%
	%	\vspace*{10\p@}%
	%	{\parindent \z@ \raggedright \normalfont
	%		\ifnum \c@secnumdepth >\m@ne
	%		\if@mainmatter%%%%% Fix for frontmatter, mainmatter, and backmatter 040920
	%		\DOCH
	%		\fi
	%		\fi
	%		\interlinepenalty\@M
	%		\if@mainmatter%%%%% Fix for frontmatter, mainmatter, and backmatter 060424
	%		\DOTI{#1}%
	%		\else%
	%		\DOTIS{#1}%
	%		\fi
	%	}
	%}
	%% For the case \chapter*:
	%\renewcommand*{\@makeschapterhead}[1]{%
	%	\vspace*{10\p@}%
	%	{\parindent \z@ \raggedright
	%		\normalfont
	%		\interlinepenalty\@M
	%		\DOTIS{#1}
	%		\vskip 100\p@
	%	}
	%}
	%\makeatother
	% Referência: http://tex.stackexchange.com/questions/13357/fncychap-package-reduce-vertical-gap-space-between-header-and-chapter-heading
	% % % % % % % % % % % % % % % % % %
	% BATATA 
	%{section}{Abstract} 
	%\addtocontents{toc}{section}{Abstract}
	\chapter*{\@Resumo}% *************** Editado FHZ
	\vspace{-12ex}
	\thispagestyle{empty}
	\pagestyle{empty}
	\iftoggle{toggleVersaoFinal}{\afterpage{\cleardoublepage}}{} % Segunda página do lado ímpar/direito
	
}{%
	\par\vspace*{5mm}
	\noindent{\@stringPalavraschave}\palavrachave
}
%%%% ========================= Título do Resumo - Fim

%%%%% ========================= Título do Abstract - Inicio
\newenvironment{foreignabstract}{%
	\chapter*{\@Abstract}% *************** Editado FHZ
	\vspace{-12ex}
	\thispagestyle{empty}
	\pagestyle{empty}
	\iftoggle{toggleVersaoFinal}{\afterpage{\cleardoublepage}}{} % Segunda página do lado ímpar/direito
}{%
	\par\vspace*{5mm}
	\noindent{\@stringKeywords}\keywordeng\
	\global\let\@author\@empty
	\global\let\@date\@empty
	\global\let\foreign@title\@empty
	\global\let\foreign@title\relax
	\global\let\local@title\@empty
	\global\let\local@title\relax
	\global\let\author\relax
	\global\let\author\relax
	\global\let\date\relax
}
%%%%% ========================= Título do Abstract - Fim
%%%%%%%%%%%%%%%% - Novos ambientes com  nome de capítulo definido - Fim

%%%%% ========================= Título da Citação Direta - Inicio
\newenvironment{citacaoDireta}{%
	\begin{flushleft}
		\small\singlespace\setlength{\leftskip}{4cm}
	\end{flushleft}
}
%%%%% ========================= Título da Citação Direta  - Fim

%%%%% ========================= Lista de Figuras - Inicio
\renewcommand\listoffigures{%
	\renewcommand\listfigurename{\@ListaIlustracoes} % BATATA - Ilustrações
	\ufabcFHZh@hasLof
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \iftoggle{toggleVersaoFinal}{\cleardoublepage}{\clearpage}
    %\cleardoublepage % BATATA - impressão
    \chapter*{\listfigurename}%
    \thispagestyle{empty}
    \pagestyle{empty}
	  %% comentado para não aparecer no Sumário: \addcontentsline
      %% \addcontentsline{toc}{chapter}{\listfigurename}%
      \@mkboth{\MakeUppercase\listfigurename}%
              {\MakeUppercase\listfigurename}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
}
%%%%% ========================= Lista de Figuras - Fim

%%%%% ========================= Lista de Tabelas - Inicio
\renewcommand\listoftables{%
	\if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename}%
    \thispagestyle{empty} 
    \pagestyle{empty}
      % comentado para não aparecer no Sumário: \addcontentsline
      %%\addcontentsline{toc}{chapter}{\listtablename}%
      \@mkboth{%
          \MakeUppercase\listtablename}%
         {\MakeUppercase\listtablename}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }
%%%%% ========================= Lista de Tabelas - Fim

%%%%% ========================= Lista de símbolos - Inicio
\newcommand\printlosymbols{%
	\renewcommand\glossaryname{\listsymbolname}%
	\@input@{\jobname.los}
}
\def\makelosymbols{%
	\newwrite\@losfile
	\immediate\openout\@losfile=\jobname.syx
	\def\symbl{\@bsphack\begingroup
		\@sanitize
		\@wrlos}\typeout
	{Writing index of symbols file \jobname.syx}%
	\let\makelosymbols\@empty
}
\@onlypreamble\makelosymbols
%%%%% ------------- Lista de símbolos - Link para página - Início
\AtBeginDocument{%
	\@ifpackageloaded{hyperref}{%
		\def\@wrlos#1#2{%
			\protected@write\@losfile{}%
			% Linha que exibe com p. #página
			{\string\indexentry{[#1] #2 \dotfill |hyperpage}{\thepage}}%
			\endgroup
			\@esphack}
	}{%
		\def\@wrlos#1#2{%
			\protected@write\@losfile{}%
			{\string\indexentry{[#1] #2 \dotfill}{\thepage}}%
			\endgroup
			\@esphack}
	}
}
%%%%% ------------- Lista de símbolos - Link para página - Fim
%%%%% ========================= Lista de símbolos - Fim

%%%%% ========================= Lista de abreviações - Inicio
\newcommand\printloabbreviations{%
	\renewcommand\glossaryname{\listabbreviationname}%
	\@input@{\jobname.lab}
}
\def\makeloabbreviations{%
	\newwrite\@labfile
	\immediate\openout\@labfile=\jobname.abx
	\def\abbrev{\@bsphack\begingroup
		\@sanitize
		\@wrlab}\typeout
	{Writing index of abbreviations file \jobname.abx}%
	\let\makeloabbreviations\@empty
}
\@onlypreamble\makeloabbreviations
%%%%% ------------- Lista de abreviações - Link para página - Início
\AtBeginDocument{%
	\@ifpackageloaded{hyperref}{%
		\def\@wrlab#1#2{%
			\protected@write\@labfile{}%
			{\string\indexentry{[#1] #2 \dotfill |hyperpage}{\thepage}}%
			\endgroup
			\@esphack}
	}{%
		\def\@wrlab#1#2{%
			\protected@write\@labfile{}%
			{\string\indexentry{[#1] #2 \dotfill}{\thepage}}%
			\endgroup
			\@esphack}
	}
}
%%%%% ------------- Lista de abreviações - Link para página - Fim
%%%%% ========================= Lista de abreviações - Fim

%%%%% ========================= Lista de bib - Inicio
\AtBeginDocument{%
	\ltx@ifpackageloaded{hyperref}{
		\def\ufabcFHZh@bibEnd{%
			\immediate\write\@auxout{%
				\string\newlabel{bib:end}{{}{\arabic{page}}{}{page.\arabic{page}}{}}}}%
		\def\ufabcFHZh@bibBegin{%
			\immediate\write\@auxout{%
				\string\newlabel{bib:begin}{{}{\arabic{page}}{}{page.\arabic{page}}{}}}}%
		\def\ufabcFHZh@mainBegin{%
			\immediate\write\@auxout{%
				\string\newlabel{front:pageno}{{}{\Roman{page}}{}{page.\roman{page}}{}}}}%
		\def\ufabcFHZh@hasLof{%
			\immediate\write\@auxout{%
				\string\newlabel{cat:lofflag}{{}{:~il.}{}{page.\roman{page}}{}}}}%
	}{%
		\def\ufabcFHZh@bibEnd{%
			\immediate\write\@auxout{%
				\string\newlabel{bib:end}{{}{\arabic{page}}{}}}}%
		\def\ufabcFHZh@bibBegin{%
			\immediate\write\@auxout{%
				\string\newlabel{bib:begin}{{}{\arabic{page}}{}}}}%
		\def\ufabcFHZh@mainBegin{%
			\immediate\write\@auxout{%
				\string\newlabel{front:pageno}{{}{\Roman{page}}{}}}}%
		\def\ufabcFHZh@hasLof{%
			\immediate\write\@auxout{%
				\string\newlabel{cat:lofflag}{{}{:~il.}{}}}}%
	}
}
%%%%% ========================= Lista de bib - Fim

%%%%% ========================= Lista de bib - Inicio
\newdimen\bibindent%
\setlength\bibindent{1.5em}%
\renewenvironment{thebibliography}[1]{%
	\onehalfspacing%
	\chapter*{\bibname}%
	\addcontentsline{toc}{chapter}{\bibname}%
	\ufabcFHZh@bibBegin
	\list{\@biblabel{\@arabic\c@enumiv}}%
	{\setlength{\labelwidth}{0ex}%
		\setlength{\leftmargin}{9.0ex}%
		\setlength{\itemindent}{-9.0ex}%
		\advance\leftmargin\labelsep%
		\@openbib@code%
		\usecounter{enumiv}%
		\let\p@enumiv\@empty%
		\renewcommand\theenumiv{\@arabic\c@enumiv}}%
	\sloppy%
	\clubpenalty4000%
	\@clubpenalty \clubpenalty%
	\widowpenalty4000%
	\sfcode`\.\@m
}
{\def\@noitemerr%
{\@latex@warning{Empty `thebibliography' environment}}%
\ufabcFHZh@bibEnd
\endlist
}
%%%%% ========================= Lista de bib - Fim

%%%%% ========================= Lista de glossário - Inicio
\newenvironment{theglossary}{%
	\if@twocolumn%
	\@restonecoltrue\onecolumn%
	\else%
	\@restonecolfalse%
	\fi%
	\@mkboth{\MakeUppercase\glossaryname}%
	{\MakeUppercase\glossaryname}%
	\chapter*{\glossaryname}%
	% Remove Lista de Sigla e de Abreviações do Sumário
	%\addcontentsline{toc}{chapter}{\glossaryname}
	\list{}
	{\setlength{\listparindent}{0in}%
		\setlength{\labelwidth}{1.0in}%
		\setlength{\leftmargin}{1.5in}%
		\setlength{\labelsep}{0.5in}%
		\setlength{\itemindent}{0in}}%
	\sloppy
}%
{\if@restonecol\twocolumn\fi \endlist}
%%%%% ========================= Lista de glossário - Fim

%%%%% ========================= Lista de index - Inicio
\renewenvironment{theindex}{%
	\if@twocolumn
	\@restonecolfalse
	\else
	\@restonecoltrue
	\fi
	\twocolumn[\@makeschapterhead{\indexname}]%
	\@mkboth{\MakeUppercase\indexname}%
	{\MakeUppercase\indexname}%
	\thispagestyle{empty}\parindent\z@
	\addcontentsline{toc}{chapter}{\indexname}
	\parskip\z@ \@plus .3\p@\relax
	\columnseprule \z@
	\columnsep 35\p@
	\let\item\@idxitem
}
{\if@restonecol\onecolumn\else\clearpage\fi}
%%%%% ========================= Lista de index - Fim
  
%%%%% ========================= Comandos personalizados - Início
%%%%%%%% ---- Abertura direita na Versão Final, qualquer uma na rascunho - Início
\newcommand{\@toogleOpenRightTrueOrFalse}{%
	% Fonte principal: http://tex.stackexchange.com/questions/213514/how-to-get-openright-only-at-frontmatter
	% Fonte auxiliar: http://tex.stackexchange.com/questions/266780/allow-chapters-to-start-on-an-even-page-in-front-matter-only
	\iftoggle{toggleVersaoFinal}{%
		\makeatletter \@openrighttrue \makeatother
	}{
		\makeatletter \@openrightfalse \makeatother
	}
}
%%%%%%%% ---- Abertura direita na Versão Final, qualquer uma na rascunho - Fim

%%%%%%%% ---- Geração de nomes de Orientador e Coorientador para folha de rosto e ficha catalográfica - Início
\newcommand{\@escritaNomesOrientadorCoorientador}{%
	{\iftoggle{toggleENTrue}{\foreign@advisorstring}{\local@advisorstring}}:&
	% BATATA - Trecho delicado para ser alterado
	\count1=0
	\toks@={}
	\@whilenum \count1<1 \do{%
		\toks@=\expandafter{%
			\csname CoppeAdvisorTitle:\the\count1%
			\expandafter\endcsname\expandafter\space%
			\csname CoppeAdvisorName:\the\count1%
			\expandafter\endcsname\expandafter\space%
			\csname CoppeAdvisorSurname:\the\count1\endcsname\\}
		\advance\count1 by 1}
	\the\toks@
	% BATATA - Copiando para coorientador antes de mudar
	\ifthenelse{\@advisor>1}{%
		{\iftoggle{toggleENTrue}{\foreign@coadvisorstring}{\local@coadvisorstring}}:&
			\toks@={}  
			\@whilenum \count1<2 \do{% BATATA - Linha antiga: \@whilenum \count1<\@advisor \do{%
				\toks@=\expandafter{%
					\csname CoppeAdvisorTitle:\the\count1%
					\expandafter\endcsname\expandafter\space%
					\csname CoppeAdvisorName:\the\count1%
					\expandafter\endcsname\expandafter\space%
					\csname CoppeAdvisorSurname:\the\count1\endcsname\\}
				\advance\count1 by 1}
			\the\toks@
		}{\,}
}
%%%%%%%% ---- Geração de nomes de Orientador e Coorientador para folha de rosto e ficha catalográfica - Fim

%%%%%%%% ---- Linhas de códigos originais de coopetex para a geração de nomes de Orientador e Coorientador para folha de rosto e ficha catalográfica - Linhas conservadas pelo conhecimento que elas detém - Início
\newcommand{\@escritaNomesOrientadorCoorientadorOldMethodCoppetex}{%
	\local@advisorstring &
	\count1=0
	\toks@={}
	\@whilenum \count1<\@advisor \do{%
		\ifcase\count1 % same as \ifnum0=\count1
		\toks@=\expandafter{%
			\csname CoppeAdvisorTitle:\the\count1%
			\expandafter\endcsname\expandafter\space%
			\csname CoppeAdvisorName:\the\count1%
			\expandafter\endcsname\expandafter\space%
			\csname CoppeAdvisorSurname:\the\count1\endcsname\\}
		\else
		\toks@=\expandafter\expandafter\expandafter{%
			\expandafter\the\expandafter\toks@%
			\expandafter&\expandafter\space%
			\csname CoppeAdvisorTitle:\the\count1%
			\expandafter\endcsname\expandafter\space%
			\csname CoppeAdvisorName:\the\count1\expandafter\endcsname%
			\expandafter\space\csname CoppeAdvisorSurname:\the\count1\endcsname\\
		}%
		\fi
		\advance\count1 by 1}
	\the\toks@
}
%%%%%%%% ---- Linhas de códigos originais de coopetex para a geração de nomes de Orientador e Coorientador para folha de rosto e ficha catalográfica - Linhas conservadas pelo conhecimento que elas detém - Fim
%%%%% ========================= Comandos personalizados - Fim

%%%% ========================= Dados do pdf gerado - Inicio
\newcommand\ufabcFHZh@hypersetup{%
	\begingroup
	% changes to \toks@ and \count@ are kept local;
	% it's not necessary for them, but it is usually the case
	% for \count1, because the first ten counters are written
	% to the DVI file, thus you got lucky because of PDF output
	\toks@={}% in this special case not necessary
	\count@=0 %
	\@whilenum\count@<\value{keywords}\do{%
		% * a keyword separator is not necessary,
		%    if there is just one keyword
		% * \csname CoppeKeyword:\the\count@\endcsname must be expanded
		%    at least once, to get rid of the loop depended \count@
		\ifcase\count@ % same as \ifnum0=\count@
		\toks@=\expandafter{\csname CoppeKeyword:\the\count@\endcsname}%
		\else
		\toks@=\expandafter\expandafter\expandafter{%
			\expandafter\the\expandafter\toks@
			\expandafter;\expandafter\space
			\csname CoppeKeyword:\the\count@\endcsname
		}%
		\fi
		\advance\count@ by 1 %
	}%
	\edef\x{\endgroup
		\noexpand\hypersetup{%
			pdfkeywords={\the\toks@}%
		}%
	}%
	\x
	\hypersetup{%
		pdfauthor={{\@authname} {\@authsurn}},
		pdftitle={\local@title},
		%  pdfsubject={\local@doctype\ de \@degreename\ em \local@deptname\ da COPPE/UFRJ},
		pdfsubject={\@pdfsubject},
		pdfcreator={LaTeX with CoppeTeX toolkit by ufabcFHZh},
		breaklinks={true},
		raiselinks={true},
		pageanchor={true},
	}
}
%%%% ========================= Dados do pdf gerado - Fim

%% ----------------------------------------------------------
%% End of file `ufabcFHZh.cls'.